#%%
import torch
import numpy as np
import os
import sys

# 确保能导入本地的 inference.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import ModelRunner

# --- 归一化辅助函数 ---
def normalize(data, stats):
    """将数据归一化到 [-1, 1] 区间"""
    if 'p02' in stats:
        min_val, max_val = np.array(stats['p02']), np.array(stats['p98'])
    else:
        min_val, max_val = np.array(stats['min']), np.array(stats['max'])

    span = max_val - min_val
    # 防止除以零
    span[span < 1e-8] = 1.0
    
    mid = (min_val + max_val) / 2.0
    return np.clip(2.0 * (data - mid) / span, -1.0, 1.0)

def unnormalize(data, stats):
    """将模型输出的 [-1, 1] 数据还原回真实物理单位"""
    if 'p02' in stats:
        min_val, max_val = np.array(stats['p02']), np.array(stats['p98'])
    else:
        min_val, max_val = np.array(stats['min']), np.array(stats['max'])

    span = max_val - min_val
    mid = (min_val + max_val) / 2.0
    return (data * span / 2.0) + mid

def main():
    # 1. 配置路径 (请修改为你实际的 checkpoint 路径)
    checkpoint_dir = "/home/lifan/Documents/GitHub/minbc/outputs/test_00002"
    gpu_id = 0

    # 2. 初始化模型
    print(f"Loading model from {checkpoint_dir}...")
    runner = ModelRunner(checkpoint_dir, action_dim=5, gpu_id=gpu_id)
    obs_horizon = runner.config.dp.obs_horizon
    print(f"Model Observation Horizon: {obs_horizon}")

    # 用户提供的 17维 观测数组
    raw_obs_arr = np.array([
       0.5548,    0.4891,    0.1223,    0.2997,  294.7244, # Proprio (5)
       1.2823,    0.0822,    1.7255,    0.0254,    1.8219,   -0.3389, # Tactile Index (6)
       1.5873,   -0.0489,    0.8882,    0.4631,    0.8262,    0.2633  # Tactile Thumb (6)
], dtype=np.float32)
    
    # 3. 数据预处理
    # A. 拆分数据
    raw_proprio = raw_obs_arr[:5]
    raw_tactile = raw_obs_arr[5:]

    # B. 归一化 (Normalize)
    # 必须使用训练时的统计数据进行归一化，否则模型无法理解输入
    norm_proprio = normalize(raw_proprio, runner.stats['proprio'])
    norm_tactile = normalize(raw_tactile, runner.stats['tactile'])

    # C. 构建序列 (Batch, Horizon, Dim)
    # 因为只有一帧数据，我们需要将其复制填充到所需的 obs_horizon 长度
    input_proprio = np.tile(norm_proprio, (1, obs_horizon, 1))
    input_tactile = np.tile(norm_tactile, (1, obs_horizon, 1))

    # D. 转为 Tensor
    obs_dict = {
        'proprio': torch.from_numpy(input_proprio).float(),
        'tactile': torch.from_numpy(input_tactile).float()
    }

    # 4. 执行预测
    print("Running inference...")
    # 这里假设预测 5 步 (execution_horizon=5)
    pred_action_norm = runner.predict(obs_dict, execution_horizon=5)

    # 5. 反归一化 (Unnormalize)
    pred_action_real = unnormalize(pred_action_norm, runner.stats['action'])

    # 6. 打印结果 (强制 Float 格式)
    # suppress=True 禁止科学计数法, precision=4 保留4位小数
    np.set_printoptions(suppress=True, precision=4, floatmode='fixed')
    
    print("\n" + "="*40)
    print(f"PREDICTED ACTION CHUNK (Shape: {pred_action_real.shape})")
    print("Columns: [Idx_PLP, Idx_Bend, Thb_PLP, Thb_Bend, Z_Axis]")
    print("="*40)
    print(pred_action_real)

if __name__ == "__main__":
    main()