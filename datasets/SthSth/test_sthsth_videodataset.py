#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback
import torch

# 根据你的项目根目录调整
# 假设你在 MCR 根目录下运行：
# python test_sthsth_videodataset.py ...
# 如果不是，就把项目根目录加到 sys.path
# 把项目根目录 MCR 加入 sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))              # .../MCR/datasets/SthSth
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))    # .../MCR
sys.path.insert(0, PROJECT_ROOT)

from datasets.SthSth.video_dataset import VideoDataset
from datasets.SthSth.utils_setup import get_cfg_defaults


def build_cfg(
    train_json,
    val_json,
    labels_json,
    hdf5_path,
):
    cfg = get_cfg_defaults()
    cfg.defrost()

    # ===== Dataset basics =====
    cfg.DATASET_TYPE = "video"
    cfg.TRAIN_DATASET_NAME = "something-something"
    cfg.VAL_DATASET_NAME = "something-something"

    cfg.TRAIN_DATASET_PATH = train_json
    cfg.VAL_DATASET_PATH = val_json
    cfg.LABELS_PATH = labels_json
    cfg.VIDEOS_PATH = hdf5_path

    # ===== Model / sampling =====
    cfg.MODEL_NAME = "swin"
    cfg.NUM_FRAMES = 16
    cfg.FRAME_UNIFORM = True
    cfg.NUM_TEST_CLIPS = 1
    cfg.NUM_TEST_CROPS = 1
    cfg.STRIDE = 1

    # ===== Number of classes =====
    cfg.TOTAL_ACTIONS.ACTION = 174

    # ===== Augmentations =====
    cfg.AUGMENTATIONS.VIDEO = ["VideoResize", "VideoRandomCrop", "VideoColorJitter"]

    cfg.freeze()
    return cfg


def print_sample_info(sample, split_name="train"):
    print(f"\n===== {split_name} sample info =====")
    print("sample keys:", sample.keys())

    if "video_id" in sample:
        print("video_id:", sample["video_id"])

    if "indices" in sample:
        print("indices shape:", sample["indices"].shape if hasattr(sample["indices"], "shape") else type(sample["indices"]))
        print("indices:", sample["indices"])

    if "start_frame" in sample:
        print("start_frame:", sample["start_frame"])

    if "labels" in sample:
        print("labels keys:", sample["labels"].keys())
        if "ACTION" in sample["labels"]:
            action_label = sample["labels"]["ACTION"]
            print("ACTION label:", action_label, " value:", int(action_label))

    if "video" in sample:
        video = sample["video"]
        print("video tensor shape:", tuple(video.shape))
        print("video dtype:", video.dtype)
        print("video min/max:", float(video.min()), float(video.max()))

        # expected shape:
        # [Eval_Clips x Eval_Crops, NUM_FRAMES, 3, H, W]
        if video.ndim != 5:
            print("WARNING: unexpected video ndim =", video.ndim)
        else:
            nclips, t, c, h, w = video.shape
            print(f"parsed as: nclips={nclips}, t={t}, c={c}, h={h}, w={w}")


def try_fetch(dataset, idx, split_name):
    print(f"\n--- Fetching {split_name} idx={idx} ---")
    sample = dataset[idx]
    print_sample_info(sample, split_name=split_name)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--labels_json", type=str, required=True)
    parser.add_argument("--hdf5_path", type=str, required=True)
    parser.add_argument("--train_idx", type=int, default=0)
    parser.add_argument("--val_idx", type=int, default=0)
    args = parser.parse_args()

    print("Building config...")
    cfg = build_cfg(
        train_json=args.train_json,
        val_json=args.val_json,
        labels_json=args.labels_json,
        hdf5_path=args.hdf5_path,
    )

    print("Creating train dataset...")
    train_dataset = VideoDataset(cfg, train=True)
    print("Creating val dataset...")
    val_dataset = VideoDataset(cfg, train=False)

    print("\n===== dataset sizes =====")
    print("train len:", len(train_dataset))
    print("val len:", len(val_dataset))

    # 先打印一下原始 annotation 的第一个元素，辅助检查 json 格式
    print("\n===== raw annotation examples =====")
    print("train first item:", train_dataset.dataset[0])
    print("val first item:", val_dataset.dataset[0])

    # 抓几个样本试试
    try:
        try_fetch(train_dataset, args.train_idx, "train")
    except Exception as e:
        print("\n[ERROR] train sample failed:")
        traceback.print_exc()

    try:
        try_fetch(val_dataset, args.val_idx, "val")
    except Exception as e:
        print("\n[ERROR] val sample failed:")
        traceback.print_exc()

    # 再测一个 DataLoader batch
    print("\n===== dataloader batch test =====")
    try:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(train_loader))
        print("batch keys:", batch.keys())
        print("batch['video'].shape:", tuple(batch["video"].shape))
        print("batch['labels']['ACTION'].shape:", tuple(batch["labels"]["ACTION"].shape))
        print("batch labels:", batch["labels"]["ACTION"])
    except Exception:
        print("\n[ERROR] dataloader batch test failed:")
        traceback.print_exc()


if __name__ == "__main__":
    main()