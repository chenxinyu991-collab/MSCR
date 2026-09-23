#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pack extracted video frames into HDF5.

Expected input structure:
all_video_frames_path/
  1/
    000001.jpg
    000002.jpg
    ...
  2/
    000001.jpg
    ...

Each frame is stored as one variable-length uint8 item in HDF5:
hdf5["1"][0] -> bytes of first frame
compatible with:
Image.open(io.BytesIO(hdf5["1"][idx]))
"""

import os
import json
import argparse
import h5py
import numpy as np
from pathlib import Path
from natsort import natsorted
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


def pack_video_frames_to_hdf5(args):
    frames_root = Path(args.all_video_frames_path)
    assert frames_root.exists(), f"Not found: {frames_root}"

    target_ids = None
    if args.only_from_json:
        target_ids = collect_ids_from_json([args.train_json, args.val_json])
        print(f"Collected {len(target_ids)} unique video ids from json files.")

    video_ids = [p.name for p in frames_root.iterdir() if p.is_dir()]
    video_ids = natsorted(video_ids)

    if target_ids is not None:
        video_ids = [vid for vid in video_ids if vid in target_ids]

    print(f"Packing {len(video_ids)} videos into {args.save_hdf5_path}")

    # variable-length uint8 dtype
    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))

    with h5py.File(args.save_hdf5_path, args.hdf5_mode) as hdf5_file:
        for video_id in tqdm(video_ids, desc="Packing HDF5"):
            video_frames_path = frames_root / video_id
            frame_files = [p for p in video_frames_path.iterdir() if p.is_file()]
            frame_files = natsorted(frame_files)

            if len(frame_files) == 0:
                continue

            if video_id in hdf5_file:
                if args.skip_existing:
                    continue
                else:
                    del hdf5_file[video_id]

            dset = hdf5_file.create_dataset(
                video_id,
                shape=(len(frame_files),),
                dtype=vlen_uint8,
            )

            for i, frame_path in enumerate(frame_files):
                with open(frame_path, "rb") as img_f:
                    binary_data = img_f.read()
                dset[i] = np.frombuffer(binary_data, dtype=np.uint8)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Pack video frames into HDF5.")
    parser.add_argument(
        "--all_video_frames_path",
        type=str,
        required=True,
        help="Path containing per-video frame folders.",
    )
    parser.add_argument(
        "--save_hdf5_path",
        type=str,
        required=True,
        help="Where to save the HDF5 file.",
    )
    parser.add_argument("--train_json", type=str, default=None,
                        help="Official something-something-v2-train.json")
    parser.add_argument("--val_json", type=str, default=None,
                        help="Official something-something-v2-validation.json")
    parser.add_argument("--only_from_json", action="store_true",
                        help="Only pack ids appearing in train/val json")
    parser.add_argument("--hdf5_mode", type=str, default="a",
                        choices=["a", "w"],
                        help="HDF5 open mode. Use 'w' for fresh creation.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip video ids already existing in HDF5")
    args = parser.parse_args()

    pack_video_frames_to_hdf5(args)


if __name__ == "__main__":
    main()