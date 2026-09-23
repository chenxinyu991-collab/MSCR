import os
import pickle
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

# 将核心处理函数独立出来，便于多进程调用
# 注意：多进程函数必须是顶层函数，不能嵌套在其他函数里
def process_single_video(args):
    """
    args: (video_path, label)
    """
    video_path, label = args
    audio_fps = 16000

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # 如果无法打开，返回 None，后面统一过滤
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 防止除零错误
        if frame_fps <= 0:
            frame_fps = 25 # 默认 fallback

        # 提取音频
        try:
            video_clip = VideoFileClip(video_path)
            if video_clip.audio is None:
                audio_clip = np.empty(shape=1)
            else:
                audio_clip = video_clip.audio.to_soundarray(fps=audio_fps)[:, 0]
            video_clip.close() # 记得关闭
        except Exception as e:
            # 音频提取失败也容错处理
            audio_clip = np.empty(shape=1)

        frames, frame_number = [], 0
        while frame_number < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % frame_fps == 0:
                frames.append(frame)
            frame_number += 1
        cap.release()
        frames = np.array(frames)
        
        return {
            video_path.split("/")[-1]: {
                "data": {"vision": frames, "audio": audio_clip},
                "label": label,
            }
        }
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def run_multiprocessing(data_list, label_list, desc="Processing"):
    """
    多进程处理辅助函数
    """
    # 准备参数列表
    tasks = list(zip(data_list, label_list))
    
    # 设定进程数，保守起见用 CPU 核心数的一半，最大 32
    num_workers = min(cpu_count() // 2, 32)
    # 如果核心数少，至少用 4 个
    num_workers = max(num_workers, 4)
    
    results = []
    print(f"Starting multiprocessing pool with {num_workers} workers for {desc}...")
    
    with Pool(processes=num_workers) as pool:
        # 使用 imap_unordered 可能稍微快一点，tqdm 显示进度
        for res in tqdm(pool.imap(process_single_video, tasks), total=len(tasks), desc=desc):
            if res is not None:
                results.append(res)
                
    return results

if __name__ == '__main__':
    # 路径配置
    data_roots = "/home/xinyu9981/data/ucf"

    annotation_folder = os.path.join(data_roots, "ucfTrainTestlist")
    annotation_train = os.path.join(annotation_folder, "trainlist01.txt")
    annotation_val = os.path.join(annotation_folder, "testlist01.txt")
    annotation_test = os.path.join(annotation_folder, "testlist01.txt")

    class_ind_path = os.path.join(data_roots, "ucfTrainTestlist", "classInd.txt")
    print(f"Loading class index from: {class_ind_path}")

    if not os.path.exists(class_ind_path):
        raise FileNotFoundError(f"找不到文件: {class_ind_path}")

    class_to_int = {}
    with open(class_ind_path, "r") as file:
        for line in file:
            values = line.replace("\n", "").strip().split(" ")
            class_to_int[values[1]] = values[0]

    # --- 1. 加载 Train ---
    train_data, train_labels = [], []
    with open(annotation_train, "r") as file:
        for line in file:
            values = line.replace("\n", "").strip().split(" ")
            train_data.append(os.path.join(os.path.join(data_roots, "UCF-101"), values[0]))
            train_labels.append(values[1])
    train_labels = np.array(train_labels).astype(int)

    # --- 2. 加载 Val ---
    val_data, val_labels = [], []
    with open(annotation_val, "r") as file:
        for line in file:
            line = line.replace("\n", "").strip()
            if not line: continue
            values = line.split(" ")
            video_rel_path = values[0]
            val_data.append(os.path.join(data_roots, "UCF-101", video_rel_path))
            class_name = video_rel_path.split("/")[0]
            val_labels.append(class_to_int[class_name])
    val_labels = np.array(val_labels).astype(int)

    # --- 3. 加载 Test ---
    test_data, test_labels = [], []
    with open(annotation_test, "r") as file:
        for line in file:
            line = line.replace("\n", "").strip()
            if not line: continue
            values = line.split(" ")
            video_rel_path = values[0]
            test_data.append(os.path.join(data_roots, "UCF-101", video_rel_path))
            class_name = video_rel_path.split("/")[0]
            test_labels.append(class_to_int[class_name])
    test_labels = np.array(test_labels).astype(int)

    # === 执行多进程处理 ===
    
    # Train
    raw_train_results = run_multiprocessing(train_data, train_labels, "Processing Train")
    total_train_data = {k: v for d in raw_train_results for k, v in d.items()}

    # Val
    raw_val_results = run_multiprocessing(val_data, val_labels, "Processing Val")
    total_val_data = {k: v for d in raw_val_results for k, v in d.items()}

    # Test
    raw_test_results = run_multiprocessing(test_data, test_labels, "Processing Test")
    total_test_data = {k: v for d in raw_test_results for k, v in d.items()}

    # === 合并并保存 ===
    print("Merging data...")
    total_data = {}
    total_data.update(total_train_data)
    total_data.update(total_val_data)
    total_data.update(total_test_data)

    output_pkl = os.path.join(data_roots, "UCF_fps1_audio16000.pkl")
    print(f"Saving pickle to {output_pkl} (this might take a while)...")
    
    with open(output_pkl, "wb") as handle:
        pickle.dump(total_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Done!")