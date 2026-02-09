import os
import copy
import time
import torch
import numpy as np
import torch.distributed as dist

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from dp.models.simple_act_pred import BCNetwork 
from torch import nn
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from configs.base import MinBCConfig
from utils.obs import minmax_norm_data, minmax_unnorm_data
from utils.misc import tprint, pprint


class VanillaBCPolicy:
    def __init__(
        self,
        config: MinBCConfig,
        obs_horizon,
        obs_dim,
        pred_horizon,
        action_horizon,
        action_dim,
        weight_decay=1e-6,
        learning_rate=1e-4,
        binarize_touch=False,
        device='cpu',
    ):
        self.multi_gpu = config.multi_gpu
        self.config = config
        if self.multi_gpu:
            self.rank = int(os.getenv('LOCAL_RANK', '0'))
            self.rank_size = int(os.getenv('WORLD_SIZE', '1'))
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)
            print(f'current rank: {self.rank} and use device {device}')
        else:
            self.rank = -1

        encoders = {}
        self.encoders = encoders
        self.obs_horizon = obs_horizon
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.data_stat = None
        self.writer = None
        self.binarize_touch = binarize_touch
        
        self.device = device
        self.model = BCNetwork(config).to(self.device)

        self.ema = EMAModel(parameters=self.model.parameters(), power=0.75)
        self.ema_nets = copy.deepcopy(self.model)
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def set_lr_scheduler(self, num_warmup_steps, num_training_steps):
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps,
        )

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.ema.to(device)
        self.ema_nets.to(device)

    def train(
        self,
        num_epochs,
        train_loader,
        test_loaders,
        save_path=None,
        save_freq=10,
        eval_freq=10,
        wandb_logger=None,
    ):
        if self.multi_gpu:
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        best_mse = 1e10
        best_train_mse = 1e10
        global_training_step = 0
        self.model.train()
        if self.writer is None:
            self.writer = SummaryWriter(save_path)
        
        _init_t = time.time()
        for epoch_idx in range(num_epochs):
            epoch_loss = list()
            _t = time.time()
            for data in train_loader:
                # --- FIX: Move ALL data to device ---
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device)
                
                gt_action = data["action"]
                
                # Forward pass
                pred_action = self.model(data)

                # MSE Loss
                loss = nn.functional.mse_loss(pred_action, gt_action)

                # optimize
                loss.backward()

                if self.multi_gpu:
                    all_grads_list = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset: offset + param.numel()].view_as(param.grad.data) / self.rank_size
                            )
                            offset += param.numel()

                self.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], global_training_step)
                self.writer.add_scalar(f'loss', loss.item(), global_training_step)
                global_training_step += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                self.ema.step(self.model.parameters())
                epoch_loss.append(loss.item())

            if self.rank <= 0:
                eta_s = (time.time() - _init_t) / (epoch_idx + 1) * (num_epochs - epoch_idx - 1)
                eta_h = eta_s / 3600
                tprint(f"Epoch {epoch_idx} | Loss: {np.mean(epoch_loss):.4f} | "
                       f"Epoch Time: {time.time() - _t:.2f}s | ETA: {eta_h:.1f}h")
                self.writer.add_scalar("Epoch Loss", np.mean(epoch_loss), epoch_idx)

                if wandb_logger is not None:
                    wandb_logger.step()
                    wandb_logger.log({"Epoch Loss": np.mean(epoch_loss), "epoch": epoch_idx})
                if save_path is not None and epoch_idx % save_freq == 0:
                    model_path = os.path.join(save_path, f"model_last.ckpt")
                    self.save(model_path)

            if epoch_idx % eval_freq == 0 or epoch_idx == num_epochs - 1:
                self.to_ema()
                self.ema_nets.eval()
                
                mses, normalized_mses = [], []
                for test_loader in test_loaders:
                    mse, normalized_mse = self.eval(test_loader)
                    mses.append(mse)
                    normalized_mses.append(normalized_mse)
                
                mse = mses[0]
                normalized_mse = normalized_mses[0]

                if self.rank <= 0:
                    if mse < best_mse:
                        self.save(os.path.join(save_path, f"model_best.ckpt"))
                        best_mse = mse

                    self.writer.add_scalar("Action_MSE", mse, epoch_idx)
                    self.writer.add_scalar("Normalized_MSE", normalized_mse, epoch_idx)
                    pprint(f"{self.config.output_name} | Epoch {epoch_idx} | Test MSE: {mse:.4f} | Best Test MSE: {best_mse:.4f}")

                self.ema_nets.train()

    @torch.no_grad()
    def eval(self, test_loader):
        mse = 0
        normalized_mse = 0
        cnt = 0
        for data in test_loader:
            # --- FIX: Move ALL data to device during Eval too ---
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(self.device)
            
            gt_action = data["action"]
            pred_action = self.ema_nets(data)

            pred_np = pred_action.detach().to("cpu").numpy()
            gt_np = gt_action.detach().to("cpu").numpy()
            
            action_pred = minmax_unnorm_data(pred_np, dmin=self.data_stat["action"]["min"], dmax=self.data_stat["action"]["max"])
            gt_real = minmax_unnorm_data(gt_np, dmin=self.data_stat["action"]["min"], dmax=self.data_stat["action"]["max"])
            
            _mse = mse_loss(
                torch.tensor(action_pred), torch.tensor(gt_real), reduction='none'
            ).mean(-1).mean(-1).sum()
            
            _normalized_mse = mse_loss(
                pred_action, gt_action, reduction='none'
            ).mean(-1).mean(-1).sum()
            
            mse += _mse.item()
            normalized_mse += _normalized_mse.item()
            cnt += gt_action.shape[0]

        mse /= max(1, cnt) # Prevent div by zero if empty
        normalized_mse /= max(1, cnt)
        return mse, normalized_mse

    def to_ema(self):
        self.ema.copy_to(self.ema_nets.parameters())

    def load(self, path):
        state_dict = torch.load(path, map_location="cuda", weights_only=True)
        self.ema_nets = self.model
        self.ema_nets.load_state_dict(state_dict)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.ema_nets = self.model
        torch.save(self.ema_nets.state_dict(), path)