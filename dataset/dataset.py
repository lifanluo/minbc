import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset as TorchDataset
from utils.obs import minmax_norm_data

# Define dimensions/shapes
RT_DIM = {
    "proprio_arm": 6,       # [x, y, z, rx, ry, rz]
    "proprio_hand": 4,      # [idx_bend, idx_plp, thb_bend, thb_plp]
    "nail_flow": 6,         # 3 points * 2 (u,v) = 6 flat
    "pad_flow": (2, 60, 52), 
    "rgb": (3, 480, 640),   
    "action": 10            # 6D Arm + 4D Hand
}

class Dataset(TorchDataset):
    def __init__(self, config, data_path, split='train'):
        self.config = config
        self.data_path = data_path
        self.split = split
        self.obs_horizon = config.dp.obs_horizon
        self.pre_horizon = config.dp.pre_horizon
        self.data_key = config.data.data_key 
        
        self.indices = []
        self.stats = {}
        self._load_data()

    def __len__(self):
        return len(self.indices)

    def _load_data(self):
        import glob
        # Find files recursively
        csv_files = sorted(glob.glob(os.path.join(self.data_path, "**", "log.csv"), recursive=True))

        all_data = {
            'proprio_arm': [], 'proprio_hand': [], 
            'nail_idx': [], 'nail_thb': [],
            'pad_idx': [], 'pad_thb': [],
            'action': [], 'rgb': []
        }
        
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            root_dir = os.path.dirname(csv_path)
            
            # --- A. Proprio (Inputs) ---
            arm_cols = ['ee_x_mm', 'ee_y_mm', 'ee_z_mm', 'ee_rx_deg', 'ee_ry_deg', 'ee_rz_deg']
            hand_cols = ['idx_bend', 'idx_plp', 'thb_bend', 'thb_plp']
            
            arm_data = df[arm_cols].values.astype(np.float32)
            hand_data = df[hand_cols].values.astype(np.float32)
            
            # --- B. Action (Outputs) ---
            # Concatenate Full Arm (6D) + Hand (4D) = 10D
            action_data = np.concatenate([arm_data, hand_data], axis=1)

            # --- C. Image/Flow Paths ---
            pad_idx_paths = [os.path.join(root_dir, f) for f in df['index_pad_flow_file']]
            pad_thb_paths = [os.path.join(root_dir, f) for f in df['thumb_pad_flow_file']]
            nail_idx_paths = [os.path.join(root_dir, f) for f in df['index_nail_flow_file']]
            nail_thb_paths = [os.path.join(root_dir, f) for f in df['thumb_nail_flow_file']]
            rgb_paths = [os.path.join(root_dir, f) for f in df['rgb_file']]

            all_data['proprio_arm'].append(arm_data)
            all_data['proprio_hand'].append(hand_data)
            all_data['action'].append(action_data)
            all_data['pad_idx'].extend(pad_idx_paths)
            all_data['pad_thb'].extend(pad_thb_paths)
            all_data['nail_idx'].extend(nail_idx_paths)
            all_data['nail_thb'].extend(nail_thb_paths)
            all_data['rgb'].extend(rgb_paths)
            
            # Create Indices
            episode_len = len(df)
            start_idx = len(self.indices)
            for i in range(episode_len):
                 # Prediction horizon check
                 if i + self.pre_horizon <= episode_len:
                     # (global_start, global_end)
                     self.indices.append((start_idx + i, start_idx + i + self.pre_horizon))

        # 3. Concatenate & Normalize
        self.buffer = {
            'proprio_arm': np.concatenate(all_data['proprio_arm']),
            'proprio_hand': np.concatenate(all_data['proprio_hand']),
            'action': np.concatenate(all_data['action']),
            'pad_idx_paths': all_data['pad_idx'],
            'pad_thb_paths': all_data['pad_thb'],
            'nail_idx_paths': all_data['nail_idx'],
            'nail_thb_paths': all_data['nail_thb'],
            'rgb_paths': all_data['rgb']
        }
        
        self.stats = self._compute_stats(self.buffer)
        for key in ['proprio_arm', 'proprio_hand', 'action']:
            self.buffer[key] = minmax_norm_data(self.buffer[key], self.stats[key]['min'], self.stats[key]['max'])

    def _compute_stats(self, buffer):
        stats = {}
        for key in ['proprio_arm', 'proprio_hand', 'action']:
            stats[key] = {
                'min': np.min(buffer[key], axis=0),
                'max': np.max(buffer[key], axis=0)
            }
        return stats

    def _load_npy(self, path, shape, downsample=False):
        """
        Safely load .npy file. 
        If downsample=True, slices [::5, ::5] to convert (296,257) -> (60,52).
        Returns zero array of 'shape' if failed.
        """
        try:
            arr = np.load(path)
            if downsample:
                arr = arr[::5, ::5]
            return arr.astype(np.float32)
        except Exception:
            return np.zeros(shape, dtype=np.float32)

    def _load_img(self, path):
        if not os.path.exists(path):
            return np.zeros((3, 480, 640), dtype=np.float32)
        try:
            img = cv2.imread(path)
            if img is None: return np.zeros((3, 480, 640), dtype=np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            return img.transpose(2, 0, 1)
        except Exception:
            return np.zeros((3, 480, 640), dtype=np.float32)

    def __getitem__(self, idx):
        global_start, global_end = self.indices[idx]
        slice_indices = range(global_start, global_end)

        batch = {}
        
        # Proprio (10D Input)
        p_arm = self.buffer['proprio_arm'][slice_indices]
        p_hand = self.buffer['proprio_hand'][slice_indices]
        batch['proprio'] = torch.tensor(np.concatenate([p_arm, p_hand], axis=-1))

        # Action (10D Output)
        batch['action'] = torch.tensor(self.buffer['action'][slice_indices])

        # Nail Flow - Raw Shape (3, 2)
        n_idx = np.stack([self._load_npy(self.buffer['nail_idx_paths'][i], (3,2)).flatten() for i in slice_indices])
        n_thb = np.stack([self._load_npy(self.buffer['nail_thb_paths'][i], (3,2)).flatten() for i in slice_indices])
        batch['nail'] = torch.tensor(np.concatenate([n_idx, n_thb], axis=-1))

        # Pad Flow - Downsampled from (296,257) to (60,52)
        # We pass downsample=True
        pad_idx = np.stack([self._load_npy(self.buffer['pad_idx_paths'][i], (60,52,2), downsample=True).transpose(2,0,1) for i in slice_indices])
        pad_thb = np.stack([self._load_npy(self.buffer['pad_thb_paths'][i], (60,52,2), downsample=True).transpose(2,0,1) for i in slice_indices])
        batch['pad_idx'] = torch.tensor(pad_idx)
        batch['pad_thb'] = torch.tensor(pad_thb)

        # RGB
        if 'rgb' in self.data_key:
            imgs = np.stack([self._load_img(self.buffer['rgb_paths'][i]) for i in slice_indices])
            batch['rgb'] = torch.tensor(imgs)

        # Slice Observation Horizon
        obs_h = self.obs_horizon
        batch['proprio'] = batch['proprio'][:obs_h]
        batch['nail'] = batch['nail'][:obs_h]
        batch['pad_idx'] = batch['pad_idx'][:obs_h]
        batch['pad_thb'] = batch['pad_thb'][:obs_h]
        if 'rgb' in batch:
            batch['rgb'] = batch['rgb'][:obs_h]
            
        return batch