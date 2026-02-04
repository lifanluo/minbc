import os
import torch
import numpy as np
import pickle
import dataclasses
from dp.agent import Agent
from configs.base import MinBCConfig

class ModelRunner:
    def __init__(self, checkpoint_dir, action_dim=5, gpu_id=0, policy_type='dp'):
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = checkpoint_dir
        self.policy_type = policy_type
        
        print(f"[ModelRunner] Loading model from {checkpoint_dir} (Type: {policy_type})...")
        
        # --- 1. Load Configuration ---
        default_config = MinBCConfig()
        
        new_data_config = dataclasses.replace(
            default_config.data,
            data_key=('proprio', 'tactile'),
            base_action_dim=action_dim,
            im_encoder='none',
            im_key=()
        )
        
        self.config = dataclasses.replace(
            default_config, 
            data=new_data_config,
            policy_type=policy_type 
        )
        
        self.agent = Agent(self.config, clip_far=False, load_img=False, device=self.device)
        
        # --- 2. Load Weights ---
        ckpt_path = os.path.join(checkpoint_dir, "model_best.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} not found.")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            load_dict = checkpoint['state_dict']
        else:
            load_dict = checkpoint

        if self.policy_type == 'dp':
            self.agent.policy.ema_nets.load_state_dict(load_dict, strict=False)
            self.agent.policy.ema_nets.eval()
        else:
            self.agent.policy.model.load_state_dict(load_dict, strict=False)
            self.agent.policy.model.eval()
        
        # --- 3. Load Stats ---
        stats_path = os.path.join(checkpoint_dir, "stats.pkl")
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        print("[ModelRunner] Normalization stats loaded.")

    def predict(self, obs_dict, execution_horizon=1):
        """
        Output: Action Chunk (numpy array) of shape (execution_horizon, dim)
        """
        batch = {}
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor):
                t = v.to(self.device)
                if len(t.shape) == 2: t = t.unsqueeze(0)
                batch[k] = t
            else:
                t = torch.from_numpy(v).to(self.device)
                if len(t.shape) == 2: t = t.unsqueeze(0)
                batch[k] = t

        with torch.no_grad():
            if self.config.policy_type == 'dp':
                # --- Diffusion Policy Inference ---
                # A. Encoder Pass
                obs_cond = self.agent.policy.ema_nets.forward_encoder(batch)
                
                # B. Noise Init
                B = 1 
                prediction_horizon = self.config.dp.pre_horizon
                action_dim = self.config.data.action_dim
                
                noisy_action = torch.randn(
                    (B, prediction_horizon, action_dim), device=self.device
                )
                
                # C. Denoising Loop
                scheduler = self.agent.policy.noise_scheduler
                
                # --- FIX: Move scheduler tensors to GPU ---
                if hasattr(scheduler, "alphas_cumprod") and scheduler.alphas_cumprod is not None:
                    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
                if hasattr(scheduler, "betas") and scheduler.betas is not None:
                    scheduler.betas = scheduler.betas.to(self.device)
                if hasattr(scheduler, "alphas") and scheduler.alphas is not None:
                    scheduler.alphas = scheduler.alphas.to(self.device)
                # ------------------------------------------

                scheduler.set_timesteps(self.agent.policy.num_diffusion_iters, device=self.device)

                for k in scheduler.timesteps:
                    if k.ndim == 0: k = k[None].to(self.device)
                    
                    noise_pred = self.agent.policy.ema_nets.forward_denoise(
                        obs_cond, sample=noisy_action, timestep=k
                    )
                    
                    noisy_action = scheduler.step(
                        model_output=noise_pred, timestep=k, sample=noisy_action
                    ).prev_sample

                pred_batch = noisy_action
                
            else:
                # --- Behavior Cloning Inference ---
                pred_batch = self.agent.policy.model(batch)
                if len(pred_batch.shape) == 2:
                    pred_batch = pred_batch.unsqueeze(1)

            # 3. Slice and Return
            start_idx = self.config.dp.obs_horizon - 1
            if start_idx < 0: start_idx = 0
            
            end_idx = start_idx + execution_horizon
            max_len = pred_batch.shape[1]
            
            if end_idx > max_len:
                end_idx = max_len
                
            if start_idx >= max_len:
                pred_chunk = pred_batch[0, -1:].cpu().numpy()
            else:
                pred_chunk = pred_batch[0, start_idx:end_idx].cpu().numpy()
            
        return pred_chunk