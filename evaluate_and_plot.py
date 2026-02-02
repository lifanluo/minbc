#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import tyro
import dataclasses
import pandas as pd  # Added for CSV handling
from tqdm import tqdm
from dataset.dataset import Dataset
from inference import ModelRunner

@dataclasses.dataclass
class PlotConfig:
    checkpoint_dir: str = "/home/lifan/Documents/GitHub/minbc/outputs/test_00002"
    test_data_path: str = "/home/lifan/Documents/GitHub/minbc/data/test"
    action_dim: int = 5
    gpu: int = 0
    # New parameter for Horizon
    execution_horizon: int = 5 

def unnormalize_data(ndata, stats):
    dmin = stats['min']
    dmax = stats['max']
    dmin = dmin.reshape(1, -1)
    dmax = dmax.reshape(1, -1)
    return ((ndata + 1) / 2) * (dmax - dmin + 1e-8) + dmin

def get_grouped_episodes(dataset):
    episode_indices = []
    current_episode = []
    for i in range(len(dataset)):
        curr_buffer_start = dataset.indices[i][0]
        if len(current_episode) == 0:
            current_episode.append(i)
        else:
            prev_idx = current_episode[-1]
            prev_buffer_start = dataset.indices[prev_idx][0]
            if curr_buffer_start == prev_buffer_start + 1:
                current_episode.append(i)
            else:
                episode_indices.append(current_episode)
                current_episode = [i]
    if len(current_episode) > 0:
        episode_indices.append(current_episode)
    return episode_indices

def main(cfg: PlotConfig):
    # --- 1. Initialize Inference Engine ---
    runner = ModelRunner(cfg.checkpoint_dir, cfg.action_dim, cfg.gpu)
    
    # --- 2. Load Data ---
    print(f"Loading dataset from {cfg.test_data_path}...")
    dataset = Dataset(
        runner.config,
        data_path=cfg.test_data_path, 
        data_key=('proprio', 'tactile'),
        split='test'
    )
    
    episode_indices = get_grouped_episodes(dataset)
    print(f"Found {len(episode_indices)} trajectories.")

    # --- 3. Run Inference with Strided Loop ---
    dim_labels = ['Idx PLP', 'Idx Bend', 'Thb PLP', 'Thb Bend', 'Z (mm)']

    for ep_num, indices in enumerate(episode_indices):
        print(f"Processing Trajectory {ep_num} (Horizon: {cfg.execution_horizon})...")
        
        gt_actions = []
        pred_actions = []
        
        # Lists to store data for CSV
        csv_pred_chunks = []
        csv_gt_chunks = []
        csv_start_frames = []
        
        # Use a while loop to jump by 'execution_horizon'
        i = 0
        pbar = tqdm(total=len(indices), leave=False)
        
        while i < len(indices):
            idx = indices[i]
            batch = dataset[idx]
            
            # A. Prepare Observation
            obs = {
                'proprio': batch['proprio'],
                'tactile': batch['tactile']
            }
            
            # B. Get Prediction Chunk
            # Ask model for 'execution_horizon' steps
            pred_chunk = runner.predict(obs, execution_horizon=cfg.execution_horizon)
            
            # C. Get Ground Truth Chunk
            # We need the same slice from the ground truth
            start_t = runner.config.dp.obs_horizon - 1
            end_t = start_t + len(pred_chunk) # Use len(pred_chunk) in case we are at the end
            
            gt_chunk = batch['action'][start_t : end_t].cpu().numpy()
            
            # --- Store Data for CSV (Unnormalized) ---
            # Unnormalize current chunks immediately for saving
            curr_pred_unnorm = unnormalize_data(pred_chunk, runner.stats['action'])
            curr_gt_unnorm = unnormalize_data(gt_chunk, runner.stats['action'])
            
            csv_pred_chunks.append(curr_pred_unnorm)
            csv_gt_chunks.append(curr_gt_unnorm)
            
            # Create an array indicating the start frame for this chunk
            # i is the current frame index relative to the start of the trajectory processing
            chunk_len = len(pred_chunk)
            csv_start_frames.append(np.full(chunk_len, i))

            # D. Append for Plotting
            # Since pred_chunk is (T, D), we can extend the list
            if len(pred_actions) == 0:
                pred_actions = pred_chunk
                gt_actions = gt_chunk
            else:
                pred_actions = np.concatenate([pred_actions, pred_chunk], axis=0)
                gt_actions = np.concatenate([gt_actions, gt_chunk], axis=0)
            
            # E. Stride Forward
            step_size = len(pred_chunk) # Usually 5, unless at end of episode
            i += step_size
            pbar.update(step_size)
            
        pbar.close()

        # --- 4. Save CSV ---
        # Concatenate all collected chunks
        all_pred_unnorm = np.concatenate(csv_pred_chunks, axis=0)
        all_gt_unnorm = np.concatenate(csv_gt_chunks, axis=0)
        all_start_frames = np.concatenate(csv_start_frames, axis=0)
        
        # Construct DataFrame with interleaved columns
        data_dict = {'start_frame': all_start_frames}
        
        for d in range(cfg.action_dim):
            # Interleave: pred_dim_X then gt_dim_X
            data_dict[f'pred_dim_{d}'] = all_pred_unnorm[:, d]
            data_dict[f'gt_dim_{d}'] = all_gt_unnorm[:, d]
            
        df = pd.DataFrame(data_dict)
        csv_save_name = f"traj_{ep_num}_h{cfg.execution_horizon}.csv"
        csv_save_path = os.path.join(cfg.checkpoint_dir, csv_save_name)
        df.to_csv(csv_save_path, index=False)
        print(f"Saved CSV: {csv_save_path}")

        # --- 5. Post-Process & Plot ---
        # Un-normalize (using the accumulated arrays for plotting)
        gt_real = unnormalize_data(gt_actions, runner.stats['action'])
        pred_real = unnormalize_data(pred_actions, runner.stats['action'])
        
        # Create Plot
        fig, axes = plt.subplots(cfg.action_dim, 1, figsize=(10, 15), sharex=True)
        # Handle case if action_dim=1 (axes is not iterable)
        if cfg.action_dim == 1: axes = [axes]
        
        plt.subplots_adjust(hspace=0.1)
        axes[0].set_title(f"Trajectory {ep_num} (Exec Horizon: {cfg.execution_horizon})", fontsize=14)
        
        for dim in range(cfg.action_dim):
            ax = axes[dim]
            ax.plot(gt_real[:, dim], color='black', linewidth=2, label='Expert')
            ax.plot(pred_real[:, dim], color='red', linestyle='--', linewidth=2, label='Policy')
            
            if dim < len(dim_labels):
                ax.set_ylabel(dim_labels[dim], fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if dim == 0: ax.legend(loc="upper right")
            if dim == cfg.action_dim - 1: ax.set_xlabel("Time Steps", fontsize=12)

        save_name = f"traj_{ep_num}_h{cfg.execution_horizon}.png"
        save_path = os.path.join(cfg.checkpoint_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved Plot: {save_path}")

if __name__ == "__main__":
    tyro.cli(main)