#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract TV-L1 optical flow from RGB frame folders.

Input structure:
rgb_root/
  1/
    000001.jpg
    000002.jpg
    ...
  2/
    ...

Output structure:
flow_root/
  1/
    000001.png
    000002.png
    ...
  2/
    ...

Each flow image is 3-channel:
  channel 0: normalized flow-x
  channel 1: normalized flow-y
  channel 2: zeros

Notes:
- number of flow frames = number of rgb frames - 1
- output saved as PNG
- this script explicitly limits internal CPU threading
"""

import os

# ===== Limit internal threading =====
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm

# Limit OpenCV threading
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


def check_tvl1_available():
    if not hasattr(cv2, "optflow"):
        return False
    if not hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        return False
    return True


def collect_ids_from_json(json_paths):
    ids = set()
    for jp in json_paths:
        if not jp:
            continue
        with open(jp, "r") as f:
            data = json.load(f)
        for item in data:
            ids.add(str(item["id"]))
    return ids


def encode_flow_to_rgb(flow, bound=20.0):
    """
    flow: H x W x 2, float32
    return: H x W x 3, uint8
    """
    flow = np.clip(flow, -bound, bound)

    fx = flow[..., 0]
    fy = flow[..., 1]

    fx = ((fx + bound) * (255.0 / (2 * bound))).astype(np.uint8)
    fy = ((fy + bound) * (255.0 / (2 * bound))).astype(np.uint8)
    fz = np.zeros_like(fx, dtype=np.uint8)

    rgb = np.stack([fx, fy, fz], axis=-1)
    return rgb


def read_gray_image(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    return img


def create_tvl1():
    return cv2.optflow.DualTVL1OpticalFlow_create()


def process_one_video(video_rgb_dir, video_flow_dir, bound=20.0, overwrite=False):
    video_flow_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = [p for p in video_rgb_dir.iterdir() if p.is_file()]
    frame_paths = natsorted(frame_paths)

    if len(frame_paths) < 2:
        return 0, "too_short"

    # skip if already exists and non-empty
    if (not overwrite) and any(video_flow_dir.iterdir()):
        return 0, "skipped_existing"

    # clear partial leftovers
    for p in video_flow_dir.iterdir():
        if p.is_file():
            p.unlink()

    tvl1 = create_tvl1()

    prev_gray = read_gray_image(frame_paths[0])
    num_saved = 0

    for i in range(1, len(frame_paths)):
        curr_gray = read_gray_image(frame_paths[i])

        flow = tvl1.calc(prev_gray, curr_gray, None)  # H x W x 2
        flow_img = encode_flow_to_rgb(flow, bound=bound)

        save_name = f"{i:06d}.png"
        save_path = video_flow_dir / save_name

        ok = cv2.imwrite(str(save_path), flow_img)
        if not ok:
            raise ValueError(f"Failed to save flow image: {save_path}")

        num_saved += 1
        prev_gray = curr_gray

    return num_saved, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_root", type=str, required=True,
                        help="Root directory of extracted RGB frames")
    parser.add_argument("--flow_root", type=str, required=True,
                        help="Root directory to save flow frame folders")
    parser.add_argument("--train_json", type=str, default=None,
                        help="Official train json")
    parser.add_argument("--val_json", type=str, default=None,
                        help="Official validation json")
    parser.add_argument("--only_from_json", action="store_true",
                        help="Only process ids appearing in json files")
    parser.add_argument("--bound", type=float, default=20.0,
                        help="Clip bound for flow encoding, default=20.0")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing flow folders")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="For debugging: only process first N videos")
    args = parser.parse_args()

    if not check_tvl1_available():
        raise RuntimeError(
            "OpenCV TV-L1 is not available.\n"
            "Please install a compatible opencv-contrib-python-headless version."
        )

    rgb_root = Path(args.rgb_root)
    flow_root = Path(args.flow_root)
    flow_root.mkdir(parents=True, exist_ok=True)

    assert rgb_root.exists(), f"Not found: {rgb_root}"

    target_ids = None
    if args.only_from_json:
        target_ids = collect_ids_from_json([args.train_json, args.val_json])
        print(f"Collected {len(target_ids)} unique ids from json files.")

    video_ids = [p.name for p in rgb_root.iterdir() if p.is_dir()]
    video_ids = natsorted(video_ids)

    if target_ids is not None:
        video_ids = [vid for vid in video_ids if vid in target_ids]

    if args.max_videos is not None:
        video_ids = video_ids[: args.max_videos]

    print(f"Processing {len(video_ids)} videos from {rgb_root}")
    print(f"Saving flow frames to {flow_root}")
    print(f"Flow clip bound = {args.bound}")
    print("Thread limits:")
    print("  OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
    print("  OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS"))
    print("  MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
    print("  cv2 threads forced to 1")

    stats = {
        "ok_videos": 0,
        "skipped_existing": 0,
        "too_short": 0,
        "failed": 0,
        "saved_flow_frames": 0,
    }

    failed_list = []

    for video_id in tqdm(video_ids, desc="Extracting TV-L1 flow"):
        video_rgb_dir = rgb_root / video_id
        video_flow_dir = flow_root / video_id

        try:
            num_saved, status = process_one_video(
                video_rgb_dir=video_rgb_dir,
                video_flow_dir=video_flow_dir,
                bound=args.bound,
                overwrite=args.overwrite,
            )

            if status == "ok":
                stats["ok_videos"] += 1
                stats["saved_flow_frames"] += num_saved
            elif status == "skipped_existing":
                stats["skipped_existing"] += 1
            elif status == "too_short":
                stats["too_short"] += 1

        except Exception as e:
            stats["failed"] += 1
            failed_list.append((video_id, str(e)))

    print("\n===== FLOW EXTRACTION SUMMARY =====")
    for k, v in stats.items():
        print(f"{k}: {v}")

    if failed_list:
        fail_log = flow_root / "flow_failed.log"
        with open(fail_log, "w") as f:
            for vid, err in failed_list:
                f.write(f"{vid}\n{err}\n{'='*80}\n")
        print(f"Failed log saved to: {fail_log}")


if __name__ == "__main__":
    main()