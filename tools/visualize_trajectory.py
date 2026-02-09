#!/usr/bin/env python3
"""Interactive viewer for MinBC trajectory frames.

Shows index_comp, thumb_comp, and rgb images with end-effector pose. Use the
slider to jump to any frame and the stream button to play/pause.
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
try:
    import tkinter as tk
    from tkinter import ttk
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

EE_COLS = [
    "ee_x_mm",
    "ee_y_mm",
    "ee_z_mm",
    "ee_rx_deg",
    "ee_ry_deg",
    "ee_rz_deg",
]

IMG_COLS = {
    "index_comp": "index_comp_file",
    "thumb_comp": "thumb_comp_file",
    "rgb": "rgb_file",
}

IMG_SUBDIR = {
    "index_comp": "index_comp",
    "thumb_comp": "thumb_comp",
    "rgb": "rgb",
}


def _load_rgb(path: str, size: Tuple[int, int] | None = None) -> np.ndarray:
    if not os.path.exists(path):
        return np.zeros((480, 640, 3), dtype=np.uint8)
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img)


def _resolve_image_path(data_dir: str, kind: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    if os.sep in value:
        return os.path.join(data_dir, value)
    return os.path.join(data_dir, IMG_SUBDIR[kind], value)


class TrajectoryViewer:
    def __init__(self, data_dir: str, fps: float = 8.0) -> None:
        self.base_dir = os.path.abspath(data_dir)
        self.traj_dirs = self._discover_trajectories(self.base_dir)
        if not self.traj_dirs:
            raise FileNotFoundError(f"No trajectory folders found in: {self.base_dir}")

        self.traj_idx = 0
        self.data_dir = self.traj_dirs[self.traj_idx]
        self.df = self._load_dataframe(self.data_dir)

        self.frame_idx = 0
        self.streaming = False

        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 5))
        plt.subplots_adjust(bottom=0.24, top=0.9, wspace=0.05)

        initial_imgs = self._load_images(0)
        self.img_handles = [
            self.axes[0].imshow(initial_imgs[0], interpolation="nearest", aspect="equal"),
            self.axes[1].imshow(initial_imgs[1], interpolation="nearest", aspect="equal"),
            self.axes[2].imshow(initial_imgs[2], interpolation="nearest", aspect="equal"),
        ]
        self.axes[0].set_title("index_comp")
        self.axes[1].set_title("thumb_comp")
        self.axes[2].set_title("rgb")
        for ax in self.axes:
            ax.set_aspect("equal")
            ax.set_adjustable("box")
            ax.axis("off")

        self.ee_text = self.fig.text(0.02, 0.95, "", fontsize=10, va="top")

        traj_ax = self.fig.add_axes([0.2, 0.13, 0.55, 0.03])
        self.traj_slider = Slider(
            traj_ax,
            "Traj",
            0,
            max(len(self.traj_dirs) - 1, 0),
            valinit=0,
            valstep=1,
        )
        self.traj_names = [os.path.basename(p) for p in self.traj_dirs]
        # display the folder name instead of numeric value
        try:
            self.traj_slider.valtext.set_text(self.traj_names[self.traj_idx])
        except Exception:
            pass
        self.traj_slider.on_changed(self._on_traj_change)

        slider_ax = self.fig.add_axes([0.2, 0.08, 0.55, 0.03])
        self.slider = Slider(
            slider_ax,
            "Frame",
            0,
            len(self.df) - 1,
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(self._on_slider_change)

        button_ax = self.fig.add_axes([0.8, 0.07, 0.12, 0.05])
        self.button = Button(button_ax, "Stream")
        self.button.on_clicked(self._toggle_stream)

        # Prev / Next trajectory buttons
        prev_ax = self.fig.add_axes([0.05, 0.13, 0.06, 0.04])
        self.prev_button = Button(prev_ax, "<<")
        self.prev_button.on_clicked(lambda _e: self._change_traj(-1))

        next_ax = self.fig.add_axes([0.13, 0.13, 0.06, 0.04])
        self.next_button = Button(next_ax, ">>")
        self.next_button.on_clicked(lambda _e: self._change_traj(1))

        # Dropdown-like selector (uses Tkinter combobox) opened by a Button
        select_ax = self.fig.add_axes([0.8, 0.13, 0.12, 0.04])
        self.select_button = Button(select_ax, "Choose Traj")
        if TK_AVAILABLE:
            self.select_button.on_clicked(lambda _e: self._open_traj_selector())
        else:
            # fallback to cycling
            self.select_button.on_clicked(lambda _e: self._change_traj(1))
        interval_ms = int(1000 / max(fps, 0.1))
        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self._on_timer)

        self.update_frame(0)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [col for col in EE_COLS if col not in df.columns]
        missing += [col for col in IMG_COLS.values() if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in log.csv: {missing}")

    def _discover_trajectories(self, base_dir: str) -> list[str]:
        if os.path.basename(base_dir) == "log.csv":
            base_dir = os.path.dirname(base_dir)
        if os.path.isfile(base_dir) and base_dir.endswith("log.csv"):
            base_dir = os.path.dirname(base_dir)

        if os.path.exists(os.path.join(base_dir, "log.csv")):
            return [base_dir]

        trajs = []
        for name in sorted(os.listdir(base_dir)):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "log.csv")):
                trajs.append(path)
        return trajs

    def _load_dataframe(self, data_dir: str) -> pd.DataFrame:
        csv_path = os.path.join(data_dir, "log.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"log.csv not found: {csv_path}")
        df = pd.read_csv(csv_path)
        self._validate_columns(df)
        return df

    def _on_slider_change(self, value: float) -> None:
        idx = int(value)
        if idx != self.frame_idx:
            self.update_frame(idx)

    def _on_traj_change(self, value: float) -> None:
        idx = int(value)
        if idx == self.traj_idx:
            return
        self.traj_idx = idx
        self.data_dir = self.traj_dirs[self.traj_idx]
        self.df = self._load_dataframe(self.data_dir)
        self.frame_idx = 0
        self.slider.valmax = max(len(self.df) - 1, 0)
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
        self.slider.set_val(0)
        # update displayed traj name
        try:
            self.traj_slider.valtext.set_text(self.traj_names[self.traj_idx])
        except Exception:
            pass

    def _toggle_stream(self, _event) -> None:
        self.streaming = not self.streaming
        self.button.label.set_text("Pause" if self.streaming else "Stream")
        if self.streaming:
            self.timer.start()
        else:
            self.timer.stop()

    def _on_timer(self) -> None:
        if not self.streaming:
            return
        next_idx = (self.frame_idx + 1) % len(self.df)
        self.slider.set_val(next_idx)

    def _change_traj(self, delta: int) -> None:
        """Change trajectory by delta (can be +/-1)."""
        new_idx = (self.traj_idx + delta) % len(self.traj_dirs)
        if new_idx == self.traj_idx:
            return
        self.traj_idx = new_idx
        self.data_dir = self.traj_dirs[self.traj_idx]
        self.df = self._load_dataframe(self.data_dir)
        # update sliders
        self.slider.valmax = max(len(self.df) - 1, 0)
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
        self.traj_slider.set_val(self.traj_idx)
        self.slider.set_val(0)
        try:
            self.traj_slider.valtext.set_text(self.traj_names[self.traj_idx])
        except Exception:
            pass

    def _set_traj_by_name(self, name: str) -> None:
        for i, p in enumerate(self.traj_dirs):
            if os.path.basename(p) == name or p == name:
                self.traj_idx = i
                self.data_dir = self.traj_dirs[self.traj_idx]
                self.df = self._load_dataframe(self.data_dir)
                self.slider.valmax = max(len(self.df) - 1, 0)
                self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
                self.traj_slider.set_val(self.traj_idx)
                self.slider.set_val(0)
                try:
                    self.traj_slider.valtext.set_text(self.traj_names[self.traj_idx])
                except Exception:
                    pass
                return

    def _open_traj_selector(self) -> None:
        """Open a small Tkinter combobox window to choose a trajectory."""
        if not TK_AVAILABLE:
            return
        root = tk.Tk()
        root.withdraw()
        win = tk.Toplevel(root)
        win.title("Select Trajectory")
        ttk.Label(win, text="Trajectory:").pack(padx=8, pady=4)
        names = [os.path.basename(p) for p in self.traj_dirs]
        var = tk.StringVar(value=names[self.traj_idx])
        combo = ttk.Combobox(win, values=names, textvariable=var, state="readonly")
        combo.pack(padx=8, pady=4)

        def on_ok():
            sel = combo.get()
            if sel in names:
                self._set_traj_by_name(sel)
            win.destroy()
            try:
                root.quit()
            except Exception:
                pass

        btn = ttk.Button(win, text="OK", command=on_ok)
        btn.pack(padx=8, pady=6)
        # Make the dialog modal
        win.transient(root)
        win.grab_set()
        root.deiconify()
        root.lift()
        root.mainloop()
        root.destroy()

    def _load_images(self, idx: int) -> list[np.ndarray]:
        row = self.df.iloc[idx]
        img_paths = [
            _resolve_image_path(self.data_dir, "index_comp", row[IMG_COLS["index_comp"]]),
            _resolve_image_path(self.data_dir, "thumb_comp", row[IMG_COLS["thumb_comp"]]),
            _resolve_image_path(self.data_dir, "rgb", row[IMG_COLS["rgb"]]),
        ]
        return [_load_rgb(path) for path in img_paths]

    def update_frame(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, len(self.df) - 1))
        self.frame_idx = idx

        row = self.df.iloc[idx]
        imgs = self._load_images(idx)

        for handle, img in zip(self.img_handles, imgs):
            handle.set_data(img)

        ee_vals = row[EE_COLS].to_numpy(dtype=float)
        ee_text = (
            f"EE pose (mm/deg) | x: {ee_vals[0]:.2f}, y: {ee_vals[1]:.2f}, "
            f"z: {ee_vals[2]:.2f}, rx: {ee_vals[3]:.2f}, "
            f"ry: {ee_vals[4]:.2f}, rz: {ee_vals[5]:.2f}"
        )
        self.ee_text.set_text(ee_text)
        self.fig.canvas.draw_idle()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MinBC trajectory frames.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/lifan/Documents/GitHub/minbc/data",
        help="Path to a trajectory folder or the parent data directory",
    )
    parser.add_argument("--fps", type=float, default=8.0, help="Stream FPS")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = TrajectoryViewer(args.data_dir, fps=args.fps)
    plt.show()


if __name__ == "__main__":
    main()
