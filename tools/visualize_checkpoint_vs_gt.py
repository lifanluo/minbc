import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.base import MinBCConfig
from dataset import data_processing
from dp.agent import Agent


def merge_dataclass_and_dict(dataclass_obj: Any, update_dict: dict) -> Any:
    if not hasattr(dataclass_obj, "__dataclass_fields__"):
        raise ValueError("Provided object is not a dataclass instance.")

    dataclass_fields = fields(dataclass_obj)
    dataclass_values = {field.name: getattr(dataclass_obj, field.name) for field in dataclass_fields}
    merged_values = {}

    for key, value in dataclass_values.items():
        if key in update_dict:
            if hasattr(value, "__dataclass_fields__"):
                merged_values[key] = merge_dataclass_and_dict(value, update_dict[key])
            else:
                merged_values[key] = update_dict[key]
        else:
            merged_values[key] = value

    return dataclass_obj.__class__(**merged_values)


def _prepare_obs_step(step: Dict[str, Any], data_key: List[str]) -> Dict[str, Any]:
    obs = {}
    for key in data_key:
        if key == "img":
            if "base_rgb" not in step:
                raise KeyError("base_rgb not found in step; did you convert with --include-rgb?")
            img = step["base_rgb"]
            if img.ndim == 3:
                img = img[None, ...]
            obs[key] = img
        else:
            if key not in step:
                raise KeyError(f"Missing key '{key}' in step data")
            obs[key] = step[key]
    return obs


def _build_obs_deque(data: List[Dict[str, Any]], idx: int, obs_horizon: int, data_key: List[str]) -> List[Dict[str, Any]]:
    start = idx - obs_horizon + 1
    obs_steps = []
    for offset in range(obs_horizon):
        src_idx = max(0, start + offset)
        obs_steps.append(_prepare_obs_step(data[src_idx], data_key))
    return obs_steps


def _plot_actions(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    title: str,
    action_names: Optional[List[str]] = None,
) -> None:
    action_dim = gt.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(action_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i in range(action_dim):
        ax = axes[i]
        ax.plot(gt[:, i], label="gt", linewidth=1.5)
        ax.plot(pred[:, i], label="pred", linewidth=1.2)
        if action_names is not None and i < len(action_names):
            ax.set_title(action_names[i])
        else:
            ax.set_title(f"dim {i}")
        ax.grid(True, alpha=0.3)
    for j in range(action_dim, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    # ===== User-configurable section (edit here) =====
    run_dir = Path("/home/lifan/Documents/GitHub/minbc/outputs/card_train2")
    data_dir = Path("/home/lifan/Documents/GitHub/minbc/data/card_train")
    splits = ["train", "test"]
    ckpt = "model_best.ckpt"
    max_steps = 200
    action_index = 0  # 0 = first predicted action in horizon
    out_dir = run_dir / "vis"
    device_override = None  # e.g., "cpu" or "cuda:0"
    action_names = [
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_rx",
        "ee_ry",
        "ee_rz",
        "idx_bend",
        "idx_plp",
        "idx_knuck",
        "thb_bend",
        "thb_plp",
        "thb_knuck",
    ]
    # ===== End user-configurable section =====

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")

    with config_path.open("r") as f:
        cfg = json.load(f)
    config = merge_dataclass_and_dict(MinBCConfig(), cfg)

    data_key = list(config.data.data_key)
    load_img = "img" in data_key

    device = device_override
    if device is None:
        if torch.cuda.is_available() and config.gpu is not None:
            device = f"cuda:{config.gpu}"
        else:
            device = "cpu"

    agent = Agent(
        config,
        clip_far=False,
        num_diffusion_iters=config.dp.diffusion_iters,
        load_img=load_img,
        num_workers=0,
        dit=False,
        device=device,
    )

    ckpt_path = run_dir / ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    agent.load(str(ckpt_path))

    obs_horizon = config.dp.obs_horizon
    action_horizon = config.dp.act_horizon
    if action_index < 0 or action_index >= action_horizon:
        raise ValueError(f"action_index must be in [0, {action_horizon - 1}]")

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Skip split not found: {split_dir}")
            continue

        episodes = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        if not episodes:
            print(f"Skip empty split: {split_dir}")
            continue

        split_out_dir = out_dir / split
        split_out_dir.mkdir(parents=True, exist_ok=True)
        summary = []

        for episode_dir in episodes:
            data = data_processing.iterate(str(episode_dir), config, load_img=load_img)
            if not data:
                print(f"Skip empty episode: {episode_dir.name}")
                continue

            steps = min(len(data), max_steps)
            preds, gts = [], []
            for i in range(steps):
                obs_deque = _build_obs_deque(data, i, obs_horizon, data_key)
                pred_seq = agent.predict(obs_deque)
                pred_action = pred_seq[action_index]
                gt_action = data[i]["action"]
                preds.append(pred_action)
                gts.append(gt_action)

            preds = np.stack(preds)
            gts = np.stack(gts)

            mse = ((preds - gts) ** 2).mean(axis=0)
            plot_path = split_out_dir / f"pred_vs_gt_{episode_dir.name}.png"
            _plot_actions(
                gts,
                preds,
                plot_path,
                title=f"{episode_dir.name} | {split} | MSE avg={mse.mean():.4f}",
                action_names=action_names,
            )

            npy_path = split_out_dir / f"pred_vs_gt_{episode_dir.name}.npz"
            np.savez(npy_path, pred=preds, gt=gts, mse=mse)

            summary.append({
                "episode": episode_dir.name,
                "steps": int(steps),
                "mean_mse": float(mse.mean()),
                "plot": str(plot_path),
                "data": str(npy_path),
            })
            print(f"Saved plot: {plot_path}")

        summary_path = split_out_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
