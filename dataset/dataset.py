import os
import torch
import pickle
import numpy as np
import pandas as pd  # Added for CSV reading
import torch.utils.data
from glob import glob
from typing import Tuple, Literal, Dict, Optional
from dataset import data_processing
from configs.base import MinBCConfig
from utils.obs import minmax_norm_data

# Define dimensions for your specific robot data
RT_DIM = {
    "proprio": 5,      # ['idx_plp', 'idx_bend', 'thb_plp', 'thb_bend', 'ee_z_mm']
    "tactile": 12,     # 6D Index + 6D Thumb
    # "action" is implicitly handled (matches proprio dimension in your case)
}

def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    # Relaxed check for larger tactile values if necessary
    if np.any(stats["max"] > 1e9) or np.any(stats["min"] < -1e9):
        raise ValueError("data out of range")
    return stats


def minmax_norm_data(data, dmin, dmax):
    ndata = (data - dmin) / (dmax - dmin + 1e-8)
    ndata = ndata * 2 - 1
    return ndata


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: MinBCConfig,
        data_path: str,
        data_key: Tuple[str, ...],
        stats: dict = None,
        transform=None,
        load_img: bool = False,
        binarize_touch: bool = False,
        split: Literal["train", "test"] = "train",
        percentiles: Optional[Dict[str, float]] = None,
    ):
        self.data_key = data_key
        self.pre_horizon = config.dp.pre_horizon
        self.obs_horizon = config.dp.obs_horizon
        self.act_horizon = config.dp.act_horizon
        self.transform = transform
        self.load_img = load_img

        # --- CUSTOM DATA LOADING LOGIC ---
        all_proprio = []
        all_tactile = []
        all_actions = []
        episode_ends = []
        total_length = 0

        # Handle data_path input (can be list or string)
        root_path = data_path[0] if isinstance(data_path, list) else data_path
        
        # Find all log.csv files
        search_pattern = os.path.join(root_path, "**", "log.csv")
        csv_files = sorted(glob(search_pattern, recursive=True))
        
        print(f"[Dataset] Scanning {root_path}... Found {len(csv_files)} trajectories.")

        for csv_path in csv_files:
            folder_path = os.path.dirname(csv_path)
            idx_flow_dir = os.path.join(folder_path, 'index_nail_flow')
            thb_flow_dir = os.path.join(folder_path, 'thumb_nail_flow')
            
            try:
                # 'engine=c' is faster
                df = pd.read_csv(csv_path, engine='c')
            except Exception as e:
                print(f"Skipping {csv_path}: {e}")
                continue

            # 1. Load Proprioception (5 dims)
            req = ['idx_plp', 'idx_bend', 'thb_plp', 'thb_bend', 'ee_z_mm']
            if not all(c in df.columns for c in req):
                print(f"Skipping {csv_path}: Missing columns.")
                continue

            # (T, 5)
            proprio_data = df[req].values.astype(np.float32)
            length = len(proprio_data)
            
            # 2. Load Tactile (12 dims)
            # We must load these now to create a synchronized array
            tactile_seq = []
            
            # Pre-check if files exist to avoid crashing mid-loop
            first_idx_path = os.path.join(idx_flow_dir, "index_nail_000000.npy")
            if not os.path.exists(first_idx_path):
                print(f"Skipping {csv_path}: Tactile files not found.")
                continue

            for t in range(length):
                idx_p = os.path.join(idx_flow_dir, f"index_nail_{t:06d}.npy")
                thb_p = os.path.join(thb_flow_dir, f"thumb_nail_{t:06d}.npy")
                
                try:
                    # Assume shape is (6,) or (1,6) -> flatten to (6,)
                    idx_val = np.load(idx_p).flatten()
                    thb_val = np.load(thb_p).flatten()
                    tactile_seq.append(np.concatenate([idx_val, thb_val]))
                except FileNotFoundError:
                    # Handle case where tactile data might be shorter than CSV
                    # Pad with last frame or zeros, or cut trajectory.
                    # Here we cut the trajectory to current length to be safe.
                    proprio_data = proprio_data[:t]
                    length = t
                    break
            
            if length == 0:
                continue

            tactile_data = np.array(tactile_seq, dtype=np.float32)

            # 3. Create Action (Same as Proprioception)
            action_data = proprio_data.copy()

            all_proprio.append(proprio_data)
            all_tactile.append(tactile_data)
            all_actions.append(action_data)
            
            total_length += length
            episode_ends.append(total_length)
            print(f"Loaded trajectory: {length} steps")

        # Combine all lists into single numpy arrays
        if len(all_proprio) == 0:
            raise ValueError("No valid data found! Check paths and CSV columns.")

        train_data = {
            "data": {
                "proprio": np.concatenate(all_proprio, axis=0),
                "tactile": np.concatenate(all_tactile, axis=0),
                "action": np.concatenate(all_actions, axis=0)
            },
            "meta": {
                "episode_ends": episode_ends
            }
        }
        
        print("Dataset loaded successfully.")
        print("Proprio Shape:", train_data["data"]["proprio"].shape)
        print("Tactile Shape:", train_data["data"]["tactile"].shape)
        print("Action Shape:", train_data["data"]["action"].shape)

        # --- NORMALIZATION LOGIC ---
        # We handle 'proprio' and 'tactile' using percentile normalization.
        
        self.percentiles = {} if percentiles is None else percentiles
        
        # Only iterate over keys that exist in our loaded data
        keys_to_normalize = [k for k in self.data_key if k != "img"]

        for data_type in keys_to_normalize:
            d = train_data["data"][data_type]

            if split == "train":
                p2 = np.percentile(d, 2, axis=0)
                p98 = np.percentile(d, 98, axis=0)
                self.percentiles[data_type] = {'lower': p2, 'upper': p98}
            else:
                if data_type not in self.percentiles:
                     # Fallback if testing on data without stats (should not happen in proper training)
                     p2 = np.percentile(d, 2, axis=0)
                     p98 = np.percentile(d, 98, axis=0)
                else:
                    p2 = self.percentiles[data_type]['lower']
                    p98 = self.percentiles[data_type]['upper']

            print(f"Normalizing {data_type}...")
            # print(f"  P2: {[round(num, 4) for num in p2]}")
            # print(f"  P98: {[round(num, 4) for num in p98]}")

            mid = 0.5 * (p2 + p98)
            span = (p98 - p2)
            # Avoid divide-by-zero if a dim is constant
            eps = 1e-12
            span_safe = np.where(span < eps, 1.0, span)
            y = 2.0 * (d - mid) / span_safe  # 2–98% -> [-1, 1]
            y = np.clip(y, -1.5, 1.5)  # cap to [-1.5, 1.5]
            
            # Update the data in place
            train_data["data"][data_type] = y

        # Create indices for sampling (sliding window)
        self.indices = create_sample_indices(
            episode_ends=train_data["meta"]["episode_ends"],
            sequence_length=self.pre_horizon,
            pad_before=self.obs_horizon - 1,
            pad_after=self.act_horizon - 1,
        )

        # Normalize Action (min-max normalization to [-1, 1])
        if stats is None:
            stats = dict()
            stats["action"] = get_data_stats(train_data["data"]["action"])

        train_data["data"]["action"] = minmax_norm_data(
            train_data["data"]["action"], 
            dmin=stats["action"]["min"], 
            dmax=stats["action"]["max"]
        )

        self.stats = stats
        self.train_data = train_data["data"] # Point directly to data dict
        self.binarize_touch = binarize_touch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.pre_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        
        # Prepare output dict
        batch = {}
        
        # Handle observations
        for k in self.data_key:
            # Take only the observation horizon (first N steps of the sequence)
            obs_data = nsample[k][: self.obs_horizon]
            batch[k] = torch.tensor(obs_data, dtype=torch.float32)

        # Handle action
        batch["action"] = torch.tensor(nsample["action"], dtype=torch.float32)
        
        return batch