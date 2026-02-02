#%%
import os
import numpy as np
import pandas as pd

# ================= 配置区域 =================
# 1. 直接指定目标轨迹文件夹的完整路径
TARGET_TRAJECTORY_PATH = "/home/lifan/Documents/GitHub/minbc/data/test/490-550-300-100"

# 2. 想要查看该轨迹中的第几步 (Frame Index)
STEP_IDX = 55
# ===========================================

def main():
    # 1. 验证路径并定位 log.csv
    folder_path = TARGET_TRAJECTORY_PATH
    target_csv = os.path.join(folder_path, "log.csv")
    folder_name = os.path.basename(folder_path)

    if not os.path.exists(target_csv):
        print(f"Error: 目标路径下未找到 log.csv -> {target_csv}")
        return

    print(f"\n[已选中轨迹]: {folder_name}")
    print(f"[完整路径]: {folder_path}")

    # 2. 读取 CSV (Proprioception)
    try:
        # Using 'engine=c' for faster reading as defined in your dataset utility
        df = pd.read_csv(target_csv, engine='c')
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    if STEP_IDX >= len(df):
        print(f"Error: 步数 {STEP_IDX} 超出范围 (该轨迹共 {len(df)} 步)。")
        return

    # 提取 Proprio 5维数据: ['idx_plp', 'idx_bend', 'thb_plp', 'thb_bend', 'ee_z_mm']
    prop_cols = ['idx_plp', 'idx_bend', 'thb_plp', 'thb_bend', 'ee_z_mm']
    
    if not all(col in df.columns for col in prop_cols):
        print(f"Error: CSV 中缺少必要的列 {prop_cols}")
        return

    proprio_val = df.iloc[STEP_IDX][prop_cols].values.astype(np.float32)

    # 3. 读取 Tactile (NPY 文件)
    # Folders defined as 'index_nail_flow' and 'thumb_nail_flow'
    idx_flow_dir = os.path.join(folder_path, 'index_nail_flow')
    thb_flow_dir = os.path.join(folder_path, 'thumb_nail_flow')

    # Construct filenames using 6-digit zero padding
    idx_filename = f"index_nail_{STEP_IDX:06d}.npy"
    thb_filename = f"thumb_nail_{STEP_IDX:06d}.npy"
    
    idx_file_path = os.path.join(idx_flow_dir, idx_filename)
    thb_file_path = os.path.join(thb_flow_dir, thb_filename)

    if not (os.path.exists(idx_file_path) and os.path.exists(thb_file_path)):
        print(f"Error: 在步骤 {STEP_IDX} 找不到触觉 .npy 文件。")
        return

    # Load and flatten to match your 1D array requirement
    idx_tactile = np.load(idx_file_path).flatten()
    thb_tactile = np.load(thb_file_path).flatten()

    # 4. 格式化输出
    def fmt_arr(arr):
        return ", ".join([f"{x:>9.4f}" for x in arr])

    print("-" * 80)
    print(f"OBSERVATION (Folder: {folder_name}, Step: {STEP_IDX})")
    print("-" * 80)
    
    print("np.array([")
    print(f"    {fmt_arr(proprio_val)}, # Proprio (5)")
    print(f"    {fmt_arr(idx_tactile)}, # Tactile Index (6)")
    print(f"    {fmt_arr(thb_tactile)}  # Tactile Thumb (6)")
    print("], dtype=np.float32)")

if __name__ == "__main__":
    main()