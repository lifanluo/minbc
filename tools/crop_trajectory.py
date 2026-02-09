#!/usr/bin/env python3
"""Crop a trajectory folder by removing all frames after a given index.

This moves deleted files into a backup folder inside the trajectory directory
so deletion is reversible.

Example:
    python tools/crop_trajectory.py /home/lifan/Documents/GitHub/minbc/data/001 --last-frame 103 --dry-run
    python tools/crop_trajectory.py /home/lifan/Documents/GitHub/minbc/data/017 --last-frame 120 --yes
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from typing import List

import pandas as pd


def find_files_by_basename(root: str, basename: str) -> List[str]:
    matches = []
    for dirpath, _, files in os.walk(root):
        if basename in files:
            matches.append(os.path.join(dirpath, basename))
    return matches


def collect_files_to_remove(traj_dir: str, rows: pd.DataFrame) -> List[str]:
    files = set()
    # consider any column name containing 'file'
    file_cols = [c for c in rows.columns if "file" in c]
    for c in file_cols:
        for val in rows[c].unique():
            if not isinstance(val, str) or val == "":
                continue
            # sometimes the csv already contains subpath; just look for basename
            basename = os.path.basename(val)
            matches = find_files_by_basename(traj_dir, basename)
            if matches:
                for m in matches:
                    files.add(m)
            else:
                # also try the raw value as relative path
                candidate = os.path.join(traj_dir, val)
                if os.path.exists(candidate):
                    files.add(candidate)
    return sorted(files)


def backup_and_move(files: List[str], traj_dir: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(traj_dir, f".cropped_backup_{stamp}")
    os.makedirs(backup_dir, exist_ok=True)
    for f in files:
        # ensure file inside traj_dir
        try:
            shutil.move(f, os.path.join(backup_dir, os.path.basename(f)))
        except Exception:
            # try to preserve subfolders
            rel = os.path.relpath(f, traj_dir)
            target = os.path.join(backup_dir, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.move(f, target)
    return backup_dir


def crop_trajectory(traj_dir: str, last_frame: int, dry_run: bool = True, yes: bool = False) -> None:
    traj_dir = os.path.abspath(traj_dir)
    csv_path = os.path.join(traj_dir, "log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"log.csv not found in {traj_dir}")

    df = pd.read_csv(csv_path)
    total = len(df)
    if last_frame >= total - 1:
        print(f"Nothing to crop: last_frame {last_frame} >= last available frame {total-1}")
        return
    if last_frame < 0:
        raise ValueError("last_frame must be >= 0")

    to_remove = df.iloc[last_frame + 1 :]
    files = collect_files_to_remove(traj_dir, to_remove)

    print(f"Trajectory: {traj_dir}")
    print(f"Total frames: {total}, will keep frames 0..{last_frame} (keep {last_frame+1} frames)")
    print(f"Rows to remove: {len(to_remove)}")
    print(f"Candidate files to remove: {len(files)}")
    for p in files[:50]:
        print("  ", p)
    if len(files) > 50:
        print("  ... and", len(files) - 50, "more")

    if dry_run:
        print("Dry-run mode; no changes made. Use --yes to apply and --dry-run to preview.")
        return

    if not yes:
        ans = input("Confirm crop and move files to backup? [y/N]: ")
        if ans.lower() != "y":
            print("Aborted by user.")
            return

    # backup log.csv
    shutil.copy2(csv_path, csv_path + ".bak")

    backup_dir = backup_and_move(files, traj_dir)
    # write new csv
    new_df = df.iloc[: last_frame + 1]
    new_df.to_csv(csv_path, index=False)

    print("Crop complete.")
    print(f"Moved {len(files)} files to backup: {backup_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Crop a trajectory directory by removing frames after a given index.")
    p.add_argument("traj_dir", type=str, help="Path to trajectory folder containing log.csv")
    p.add_argument("--last-frame", type=int, required=True, help="Last frame to keep (inclusive)")
    p.add_argument("--dry-run", action="store_true", help="Don't delete; just list candidates")
    p.add_argument("--yes", action="store_true", help="Apply changes without confirmation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    crop_trajectory(args.traj_dir, args.last_frame, dry_run=args.dry_run, yes=args.yes)
