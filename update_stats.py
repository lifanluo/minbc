#%%
import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIG ---
DATA_PATH = "/home/lifan/Documents/GitHub/minbc/data/data-20260128"  # Point to your training data folder
CHECKPOINT_DIR = "/home/lifan/Documents/GitHub/minbc/outputs/test_00001" # Point to your checkpoint folder
STATS_FILE = "/home/lifan/Documents/GitHub/minbc/outputs/test_00001/stats.pkl"

def main():
    print(f"Scanning {DATA_PATH} for data...")
    
    # 1. Collect Data paths
    # (Reusing logic from your dataset.py to find files)
    csv_files = glob.glob(os.path.join(DATA_PATH, "**", "log.csv"), recursive=True)
    if not csv_files and os.path.exists(os.path.join(DATA_PATH, "log.csv")):
        csv_files = [os.path.join(DATA_PATH, "log.csv")]
    
    csv_files = sorted(list(set(csv_files)))
    print(f"Found {len(csv_files)} trajectories.")

    all_proprio = []
    all_tactile = []

    # 2. Load Data
    for csv_path in tqdm(csv_files):
        folder_path = os.path.dirname(csv_path)
        idx_flow_dir = os.path.join(folder_path, 'index_nail_flow')
        thb_flow_dir = os.path.join(folder_path, 'thumb_nail_flow')
        
        try:
            df = pd.read_csv(csv_path, engine='c')
        except Exception:
            continue
            
        # Proprio columns
        req = ['idx_plp', 'idx_bend', 'thb_plp', 'thb_bend', 'ee_z_mm']
        if not all(c in df.columns for c in req): continue
        
        proprio_data = df[req].values.astype(np.float32)
        length = len(proprio_data)
        
        # Tactile
        tactile_seq = []
        valid = True
        for t in range(length):
            idx_p = os.path.join(idx_flow_dir, f"index_nail_{t:06d}.npy")
            thb_p = os.path.join(thb_flow_dir, f"thumb_nail_{t:06d}.npy")
            try:
                idx_val = np.load(idx_p).flatten()
                thb_val = np.load(thb_p).flatten()
                tactile_seq.append(np.concatenate([idx_val, thb_val]))
            except FileNotFoundError:
                valid = False
                break
        
        if valid and len(tactile_seq) == length:
            all_proprio.append(proprio_data)
            all_tactile.append(np.array(tactile_seq, dtype=np.float32))

    # Concatenate
    proprio_arr = np.concatenate(all_proprio, axis=0)
    tactile_arr = np.concatenate(all_tactile, axis=0)
    
    print(f"Computing stats for {len(proprio_arr)} steps...")

    # 3. Calculate Percentiles (p02 and p98)
    # This matches the logic used in your dataset.py
    proprio_p02 = np.percentile(proprio_arr, 2, axis=0)
    proprio_p98 = np.percentile(proprio_arr, 98, axis=0)
    
    tactile_p02 = np.percentile(tactile_arr, 2, axis=0)
    tactile_p98 = np.percentile(tactile_arr, 98, axis=0)

    # 4. Load Existing Stats (to keep Action Min/Max)
    stats_path = os.path.join(CHECKPOINT_DIR, STATS_FILE)
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        print("Loaded existing action stats.")
    else:
        print("Warning: stats.pkl not found. Creating new.")
        stats = {}

    # 5. Update Stats Dictionary
    stats['proprio'] = {'p02': proprio_p02, 'p98': proprio_p98}
    stats['tactile'] = {'p02': tactile_p02, 'p98': tactile_p98}

    # 6. Save
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"Success! Updated stats saved to {stats_path}")
    print("Keys now available:", stats.keys())

if __name__ == "__main__":
    main()