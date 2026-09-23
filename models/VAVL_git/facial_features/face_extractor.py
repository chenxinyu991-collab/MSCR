"""
Author: Lucas Goncalves
2023

Code adapted using pieces from code by Radek Danecek

"""
"""
Modified for Stable Parallel Execution (Silent Mode)
"""
import os
import shutil
import sys
import argparse
import logging

# 屏蔽烦人的警告和日志
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("ffmpeg").setLevel(logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
# from tqdm import tqdm # 注释掉 tqdm
from models.VAVL_git.facial_features.utils.FaceVideoDataModule import TestFaceVideoDM

# ================= 配置区域 =================
ROOT_DIR = "/inspire/ssd/project/sais-scientific-models-evaluation/public/global_user/8331-huao/llj_temp/cxy/CREMA-D/"
FILE_FORMAT = ".flv"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "enet_b2_8_best.pt")
# ===========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    args = parser.parse_args()

    input_folder = os.path.join(ROOT_DIR, "VideoFlash/")
    # 依然使用独立临时文件夹
    output_folder = os.path.join(ROOT_DIR, f"faces_temp_{args.shard_id}/") 
    save_to = os.path.join(ROOT_DIR, "Face_features/")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    if not os.path.exists(save_to):
        os.makedirs(save_to, exist_ok=True)

    print(f"[Worker {args.shard_id}] Loading model...")
    model = torch.load(MODEL_PATH)
    model.classifier = torch.nn.Identity()
    model.eval()
    model = model.cuda()

    IMG_SIZE = 224
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_files = sorted(os.listdir(input_folder))
    my_files = all_files[args.shard_id::args.num_shards]
    
    print(f"[Worker {args.shard_id}] Started processing {len(my_files)} videos.")

    processed_videos = set(os.listdir(save_to))

    # 手动计数，代替 tqdm
    count = 0
    total = len(my_files)

    for ldir in my_files:
        if not ldir.endswith(FILE_FORMAT):
            continue

        filename = ldir.replace(FILE_FORMAT, "")
        target_npy = filename + ".npy"

        if target_npy in processed_videos or os.path.exists(os.path.join(save_to, target_npy)):
            count += 1
            continue

        try:
            # 每处理 10 个打印一次，证明还活着
            if count % 10 == 0:
                print(f"[Worker {args.shard_id}] Progress: {count}/{total} - Processing {filename}")

            input_video = os.path.join(input_folder, ldir)
            processed_subfolder = None
            
            # 关键：num_workers=0 必须保持
            dm = TestFaceVideoDM(
                input_video,
                output_folder,
                processed_subfolder=processed_subfolder,
                face_detector_threshold=0.96,
                batch_size=20,
                num_workers=0, 
            )
            
            # 这里可能会打印 ffmpeg 日志，我们无法在 python 层完全屏蔽
            # 只能依靠减少并发来缓解
            dm.prepare_data()
            dm.setup()
            
            processed_subfolder_name = Path(dm.output_dir).name
            extracted_folder = os.path.join(output_folder, processed_subfolder_name, filename)
            detections_folder = os.path.join(extracted_folder, "detections")
            
            if os.path.exists(detections_folder):
                image_names = sorted(os.listdir(detections_folder))
                batch_images = []
                for image_name in image_names:
                    if image_name.endswith(("png", "jpg")):
                        try:
                            image_path = os.path.join(detections_folder, image_name)
                            img = Image.open(image_path).convert('RGB')
                            batch_images.append(test_transforms(img))
                        except:
                            pass
                
                if len(batch_images) > 0:
                    batch_tensor = torch.stack(batch_images).cuda()
                    with torch.no_grad():
                        features = model(batch_tensor).cpu().numpy()
                    np.save(os.path.join(save_to, target_npy), features)

            if os.path.exists(os.path.join(output_folder, processed_subfolder_name)):
                shutil.rmtree(os.path.join(output_folder, processed_subfolder_name))
                
        except Exception as e:
            print(f"[Worker {args.shard_id}] Error on {filename}: {e}")
        
        count += 1

    print(f"[Worker {args.shard_id}] Finished!")

if __name__ == "__main__":
    main()
