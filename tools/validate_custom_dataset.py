import argparse
from pathlib import Path

from configs.base import MinBCConfig
from dataset.dataset import Dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a converted custom dataset episode.")
    parser.add_argument(
        "--episode-dir",
        type=Path,
        required=True,
        help="Path to a converted episode directory containing .pkl files.",
    )
    parser.add_argument(
        "--with-rgb",
        action="store_true",
        help="Include img in data_key when validating.",
    )
    args = parser.parse_args()

    data_key = (
        "ee_6d",
        "hand_6d",
        "index_nail_flow",
        "index_pad_flow",
        "thumb_nail_flow",
        "thumb_pad_flow",
    )
    if args.with_rgb:
        data_key = ("img",) + data_key

    config = MinBCConfig()
    ds = Dataset(
        config=config,
        data_path=[str(args.episode_dir)],
        data_key=data_key,
        stats=None,
        transform=None,
        load_img=args.with_rgb,
        split="train",
    )

    print("Dataset length:", len(ds))
    for k, v in ds.train_data.items():
        print(k, getattr(v, "shape", type(v)))


if __name__ == "__main__":
    main()
