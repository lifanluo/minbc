import argparse
import csv
import os
import pickle
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


def _read_csv_rows(csv_path: Path) -> Iterable[dict]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _load_rgb(rgb_path: Path) -> np.ndarray:
    img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"RGB file not found: {rgb_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _resize_flow(flow: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected flow shape (H,W,2), got {flow.shape}")
    h, w = flow.shape[:2]
    target_h, target_w = target_hw
    resized = cv2.resize(flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    scale_x = target_w / max(w, 1)
    scale_y = target_h / max(h, 1)
    resized[..., 0] *= scale_x
    resized[..., 1] *= scale_y
    return resized


def _load_flow(
    flow_path: Path,
    target_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    flow = np.load(str(flow_path)).astype(np.float32)
    if flow.ndim == 2 and flow.shape[-1] == 2:
        # nail flow is small (e.g., 3x2)
        return flow.reshape(-1)
    if flow.ndim == 3 and flow.shape[-1] == 2:
        if target_hw is None:
            raise ValueError("target_hw is required for pad flow resizing")
        resized = _resize_flow(flow, target_hw)
        return resized.reshape(-1)
    raise ValueError(f"Unsupported flow shape: {flow.shape} from {flow_path}")


def _parse_float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except KeyError as exc:
        raise KeyError(f"Missing column '{key}' in log.csv") from exc


def _build_obs_action(row: dict) -> Tuple[np.ndarray, np.ndarray]:
    ee = np.array(
        [
            _parse_float(row, "ee_x_mm"),
            _parse_float(row, "ee_y_mm"),
            _parse_float(row, "ee_z_mm"),
            _parse_float(row, "ee_rx_deg"),
            _parse_float(row, "ee_ry_deg"),
            _parse_float(row, "ee_rz_deg"),
        ],
        dtype=np.float32,
    )
    hand = np.array(
        [
            _parse_float(row, "idx_bend"),
            _parse_float(row, "idx_plp"),
            _parse_float(row, "idx_knuck"),
            _parse_float(row, "thb_bend"),
            _parse_float(row, "thb_plp"),
            _parse_float(row, "thb_knuck"),
        ],
        dtype=np.float32,
    )
    action = np.concatenate([ee, hand], axis=0)
    return ee, hand, action


def convert_episode(
    episode_dir: Path,
    output_dir: Path,
    include_rgb: bool,
    pad_flow_hw: Tuple[int, int],
    max_steps: Optional[int] = None,
) -> int:
    csv_path = episode_dir / "log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"log.csv not found in {episode_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, row in enumerate(_read_csv_rows(csv_path)):
        if max_steps is not None and idx >= max_steps:
            break

        ee, hand, action = _build_obs_action(row)
        data = {
            "ee_6d": ee,
            "hand_6d": hand,
            "action": action,
        }

        index_nail_path = episode_dir / "index_nail_flow" / row["index_nail_flow_file"]
        index_pad_path = episode_dir / "index_pad_flow" / row["index_pad_flow_file"]
        thumb_nail_path = episode_dir / "thumb_nail_flow" / row["thumb_nail_flow_file"]
        thumb_pad_path = episode_dir / "thumb_pad_flow" / row["thumb_pad_flow_file"]

        data["index_nail_flow"] = _load_flow(index_nail_path)
        data["index_pad_flow"] = _load_flow(index_pad_path, pad_flow_hw)
        data["thumb_nail_flow"] = _load_flow(thumb_nail_path)
        data["thumb_pad_flow"] = _load_flow(thumb_pad_path, pad_flow_hw)

        if include_rgb:
            rgb_path = episode_dir / "rgb" / row["rgb_file"]
            data["base_rgb"] = _load_rgb(rgb_path)

        out_path = output_dir / f"step_{idx:06d}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(data, f)
        count += 1

    return count


def _iter_episode_dirs(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "log.csv").exists():
            yield child


def convert_root(
    input_root: Path,
    output_root: Path,
    include_rgb: bool,
    pad_flow_hw: Tuple[int, int],
    max_steps: Optional[int],
    split: Optional[str],
) -> None:
    if (input_root / "train").exists() or (input_root / "test").exists():
        for split_name in ["train", "test"]:
            split_dir = input_root / split_name
            if not split_dir.exists():
                continue
            out_split = output_root / split_name
            for episode_dir in _iter_episode_dirs(split_dir):
                out_ep = out_split / episode_dir.name
                steps = convert_episode(
                    episode_dir,
                    out_ep,
                    include_rgb=include_rgb,
                    pad_flow_hw=pad_flow_hw,
                    max_steps=max_steps,
                )
                print(f"Converted {steps} steps: {episode_dir} -> {out_ep}")
        return

    if split is None:
        split = "train"
    out_split = output_root / split
    for episode_dir in _iter_episode_dirs(input_root):
        out_ep = out_split / episode_dir.name
        steps = convert_episode(
            episode_dir,
            out_ep,
            include_rgb=include_rgb,
            pad_flow_hw=pad_flow_hw,
            max_steps=max_steps,
        )
        print(f"Converted {steps} steps: {episode_dir} -> {out_ep}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert custom EE/hand/flow dataset into PKL episodes for MinBC."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Input root folder (contains train/test or episode folders with log.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/custom_gr1"),
        help="Output root folder to write converted PKLs.",
    )
    parser.add_argument(
        "--include-rgb",
        action="store_true",
        help="Include RGB images (base_rgb) in output PKLs.",
    )
    parser.add_argument(
        "--pad-flow-height",
        type=int,
        default=99,
        help="Target pad-flow height after resize (default: 99).",
    )
    parser.add_argument(
        "--pad-flow-width",
        type=int,
        default=86,
        help="Target pad-flow width after resize (default: 86).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on number of steps converted per episode (for quick tests).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split name when input-root directly contains episode folders (default: train).",
    )

    args = parser.parse_args()
    pad_flow_hw = (args.pad_flow_height, args.pad_flow_width)

    convert_root(
        input_root=args.input_root,
        output_root=args.output_root,
        include_rgb=args.include_rgb,
        pad_flow_hw=pad_flow_hw,
        max_steps=args.max_steps,
        split=args.split,
    )


if __name__ == "__main__":
    main()
