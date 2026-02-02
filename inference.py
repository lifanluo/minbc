import os
import torch
import numpy as np
import pickle
import dataclasses
from dp.agent import Agent
from configs.base import MinBCConfig

class ModelRunner:
    def __init__(self, checkpoint_dir, action_dim=5, gpu_id=0):
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = checkpoint_dir
        
        # --- 1. Load Configuration & Agent ---
        print(f"[ModelRunner] Loading model from {checkpoint_dir}...")
        default_config = MinBCConfig()
        
        new_data_config = dataclasses.replace(
            default_config.data,
            data_key=('proprio', 'tactile'),
            base_action_dim=action_dim,
            im_encoder='none',
            im_key=()
        )
        self.config = dataclasses.replace(default_config, data=new_data_config)
        
        self.agent = Agent(self.config, clip_far=False, load_img=False, device=self.device)
        
        # --- 2. Load Weights ---
        ckpt_path = os.path.join(checkpoint_dir, "model_best.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} not found.")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.agent.policy.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.agent.policy.model.load_state_dict(checkpoint, strict=False)
            
        self.agent.policy.ema_nets = self.agent.policy.model
        self.agent.policy.model.eval()
        
        # --- 3. Load Statistics ---
        stats_path = os.path.join(checkpoint_dir, "stats.pkl")
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        print("[ModelRunner] Normalization stats loaded.")

    def predict(self, obs_dict, execution_horizon=1):
        """
        Output: Action Chunk (numpy array) of shape (execution_horizon, dim)
        """
        clean_obs = {}
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor):
                clean_obs[k] = v.to(self.device)
                if len(clean_obs[k].shape) == 2:
                    clean_obs[k] = clean_obs[k].unsqueeze(0)

        with torch.no_grad():
            # Model Output: (1, Pred_Horizon, Dim)
            pred_batch = self.agent.policy.model(clean_obs)
            
            # Receding Horizon Control:
            # Start at the 'current' time (obs_horizon - 1)
            # Take 'execution_horizon' steps
            start_idx = self.config.dp.obs_horizon - 1
            end_idx = start_idx + execution_horizon
            
            # Ensure we don't go out of bounds of the prediction horizon
            max_horizon = pred_batch.shape[1]
            if end_idx > max_horizon:
                end_idx = max_horizon
                
            pred_chunk = pred_batch[0, start_idx:end_idx].cpu().numpy()

        print(f"[ModelRunner] Predicted action chunk shape: {pred_chunk}")
            
        return pred_chunk