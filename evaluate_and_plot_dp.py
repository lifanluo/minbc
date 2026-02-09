#%%
import matplotlib
# Force non-interactive backend to prevent Qt/XCB errors
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import os
import numpy as np
import tyro
import dataclasses
import torch
from tqdm import tqdm
from dataset.dataset import Dataset
from inference import ModelRunner

@dataclasses.dataclass
class PlotConfig:
    checkpoint_dir: str = "/home/lifan/Documents/GitHub/minbc/outputs/bc_00001"
    test_data_path: str = "/home/lifan/Documents/GitHub/minbc/data/test"
    
    # Updated to 10D (6D Arm + 4D Hand)
    action_dim: int = 10
    
    gpu: int = 0
    # BC usually uses 1 step execution
    execution_horizon: int = 1 
    policy_type: str = "bc"
    
    # Set this to True if your model was trained with RGB
    use_rgb: bool = False

def unnormalize_data(ndata, stats):
    dmin = stats['min']
    dmax = stats['max']
    # Handle broadcasting for 1D stats array
    dmin = dmin.reshape(1, -1)
    dmax = dmax.reshape(1, -1)
    return ((ndata + 1) / 2) * (dmax - dmin + 1e-8) + dmin

def _episode_root_from_path(path: str) -> str:
    if not path:
        return ""
    subdir_names = {
        "rgb",
        "index_comp",
        "thumb_comp",
        "index_raw",
        "thumb_raw",
        "index_pad_flow",
        "index_nail_flow",
        "thumb_pad_flow",
        "thumb_nail_flow",
        "xyz_npy",
    }
    parent = os.path.basename(os.path.dirname(path))
    if parent in subdir_names:
        return os.path.dirname(os.path.dirname(path))
    return os.path.dirname(path)


def _get_episode_root(dataset, buffer_idx: int) -> str:
    for key in ["rgb_paths", "pad_idx_paths", "pad_thb_paths", "nail_idx_paths", "nail_thb_paths"]:
        if hasattr(dataset, "buffer") and key in dataset.buffer and len(dataset.buffer[key]) > buffer_idx:
            return _episode_root_from_path(dataset.buffer[key][buffer_idx])
    return ""


def get_grouped_episodes(dataset):
    """Group dataset indices by episode directory.

    This avoids merging episodes when global buffer indices are contiguous.
    """
    episode_indices = []
    current_episode = []
    current_root = ""

    for i in range(len(dataset)):
        buffer_start = dataset.indices[i][0]
        root = _get_episode_root(dataset, buffer_start)
        if not current_episode:
            current_episode = [i]
            current_root = root
            continue
        if root == current_root:
            current_episode.append(i)
        else:
            episode_indices.append(current_episode)
            current_episode = [i]
            current_root = root

    if current_episode:
        episode_indices.append(current_episode)
    return episode_indices

def main(cfg: PlotConfig):
    # 1. Load Model
    runner = ModelRunner(
        cfg.checkpoint_dir, 
        cfg.action_dim, 
        cfg.gpu, 
        policy_type=cfg.policy_type,
        use_rgb=cfg.use_rgb
    )
    
    # 2. Load Dataset
    print(f"Loading Dataset from {cfg.test_data_path}...")
    test_dataset = Dataset(
        config=runner.config, 
        data_path=cfg.test_data_path, 
        split='test'
    )
    
    print(f"Loaded {len(test_dataset)} samples.")
    grouped_episodes = get_grouped_episodes(test_dataset)
    print(f"Found {len(grouped_episodes)} episodes.")
    
    # Create output dir
    save_dir = os.path.join(cfg.checkpoint_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. Evaluate
    for ep_num, episode_idxs in enumerate(tqdm(grouped_episodes)):
        gt_actions = []
        pred_actions = []
        
        for idx in episode_idxs:
            data = test_dataset[idx]
            
            # Prepare Obs: Add Batch Dim (1, T, ...)
            obs = {}
            for k, v in data.items():
                if k == 'action': continue 
                obs[k] = v.unsqueeze(0)
            
            # Predict
            action_chunk = runner.predict(obs, execution_horizon=cfg.execution_horizon)
            
            # Take first step of chunk (Open-Loop at this step)
            pred_actions.append(action_chunk[0]) 
            
            # Get GT (first step of sequence)
            gt_seq = data['action'] 
            gt_actions.append(gt_seq[0].numpy())

        gt_actions = np.array(gt_actions)
        pred_actions = np.array(pred_actions)
        
        # 4. Un-normalize
        if 'action' in runner.stats:
            gt_real = unnormalize_data(gt_actions, runner.stats['action'])
            pred_real = unnormalize_data(pred_actions, runner.stats['action'])
        else:
            gt_real = gt_actions
            pred_real = pred_actions
        
        # 5. Plot
        dim_labels = [
            'Arm X', 'Arm Y', 'Arm Z', 'Arm Rx', 'Arm Ry', 'Arm Rz',
            'Idx Bend', 'Idx PLP', 'Thb Bend', 'Thb PLP'
        ]
        
        actual_dim = gt_real.shape[1]
        fig, axes = plt.subplots(actual_dim, 1, figsize=(10, 2 * actual_dim), sharex=True)
        if actual_dim == 1: axes = [axes]
        
        plt.subplots_adjust(hspace=0.1)
        axes[0].set_title(f"Episode {ep_num} (Horizon: {cfg.execution_horizon})", fontsize=14)
        
        for dim in range(actual_dim):
            ax = axes[dim]
            ax.plot(gt_real[:, dim], color='black', linewidth=2, label='Expert')
            ax.plot(pred_real[:, dim], color='red', linestyle='--', linewidth=2, label='Policy')
            
            label = dim_labels[dim] if dim < len(dim_labels) else f"Dim {dim}"
            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if dim == 0: ax.legend(loc="upper right")
            if dim == actual_dim - 1: ax.set_xlabel("Time Steps", fontsize=12)

        save_name = f"traj_{ep_num}.png"
        print(f"Saving plot to {os.path.join(save_dir, save_name)}")
        plt.savefig(os.path.join(save_dir, save_name), dpi=100)
        plt.close()
        # Explicitly clear memory
        plt.clf()

if __name__ == "__main__":
    tyro.cli(main)