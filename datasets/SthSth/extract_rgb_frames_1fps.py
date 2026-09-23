#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract 1-fps RGB frames from Something-Something V2 .webm videos.

Output structure:
out_dir/
  1/
    000001.jpg
    000002.jpg
    ...
  2/
    000001.jpg
    ...

Supports:
- all videos in a directory
- or only video ids appearing in train/val json files
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm


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


def ffmpeg_extract_1fps(video_path, out_dir, jpg_qscale=2):
    os.makedirs(out_dir, exist_ok=True)
    out_pattern = os.path.join(out_dir, "%06d.jpg")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(video_path),
        "-vf", "fps=1",
        "-qscale:v", str(jpg_qscale),
        out_pattern,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0, result.stderr.decode("utf-8", errors="ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, required=True,
                        help="Directory containing .webm files, e.g. /path/to/SthSth")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Directory to save extracted frame folders")
    parser.add_argument("--train_json", type=str, default=None,
                        help="Official something-something-v2-train.json")
    parser.add_argument("--val_json", type=str, default=None,
                        help="Official something-something-v2-validation.json")
    parser.add_argument("--only_from_json", action="store_true",
                        help="If set, only process ids appearing in train/val json")
    parser.add_argument("--ext", type=str, default=".webm",
                        help="Video extension, default=.webm")
    args = parser.parse_args()

    video_root = Path(args.video_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    target_ids = None
    if args.only_from_json:
        target_ids = collect_ids_from_json([args.train_json, args.val_json])
        print(f"Collected {len(target_ids)} unique video ids from json files.")

    all_videos = sorted(video_root.glob(f"*{args.ext}"))
    if target_ids is not None:
        all_videos = [p for p in all_videos if p.stem in target_ids]

    print(f"Found {len(all_videos)} videos to process.")

    failed = []
    skipped = 0

    for video_path in tqdm(all_videos, desc="Extracting 1fps RGB"):
        video_id = video_path.stem
        save_dir = out_root / video_id

        # skip if already extracted and non-empty
        if save_dir.exists() and any(save_dir.iterdir()):
            skipped += 1
            continue

        ok, err = ffmpeg_extract_1fps(video_path, save_dir)
        if not ok:
            failed.append((str(video_path), err))
            # clean possibly partial outputs
            if save_dir.exists():
                for x in save_dir.iterdir():
                    x.unlink()
                save_dir.rmdir()

    print(f"Done. skipped={skipped}, failed={len(failed)}")
    if failed:
        fail_log = out_root / "extract_failed.log"
        with open(fail_log, "w") as f:
            for vp, err in failed:
                f.write(f"{vp}\n{err}\n{'='*80}\n")
        print(f"Failed log saved to: {fail_log}")


if __name__ == "__main__":
    main()