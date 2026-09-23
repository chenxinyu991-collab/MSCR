import os
import subprocess
from tqdm import tqdm

# ================= 配置区域 =================
# 请确保这些路径和你服务器上的一致
ROOT_DIR = "/inspire/ssd/project/sais-scientific-models-evaluation/public/global_user/8331-huao/llj_temp/cxy/CREMA-D"
VIDEO_DIR = os.path.join(ROOT_DIR, "VideoFlash")  # 视频源文件夹
OUTPUT_DIR = os.path.join(ROOT_DIR, "Image-01-FPS") # 输出文件夹
FPS = 1  # 每秒抽1帧
# ===========================================

def extract_frames():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 获取所有视频文件 (.flv)
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.flv')]
    videos.sort()
    
    print(f"找到 {len(videos)} 个视频，开始抽帧...")

    for video_name in tqdm(videos):
        video_path = os.path.join(VIDEO_DIR, video_name)
        file_base_name = os.path.splitext(video_name)[0]
        
        # 为每个视频创建一个子文件夹
        save_folder = os.path.join(OUTPUT_DIR, file_base_name)
        
        # 如果文件夹已经存在且不为空，跳过（断点续传）
        if os.path.exists(save_folder) and len(os.listdir(save_folder)) > 0:
            continue
            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # 构造 ffmpeg 命令
        # -i: 输入
        # -vf fps=1: 过滤器，每秒一帧
        # -q:v 2: 图片质量 (1-31, 1最高)
        # -loglevel error: 减少输出日志
        output_pattern = os.path.join(save_folder, "img_%02d.jpg")
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={FPS}",
            "-q:v", "2", 
            "-loglevel", "error",
            output_pattern
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"处理 {video_name} 时出错: {e}")
        except FileNotFoundError:
            print("错误: 找不到 ffmpeg。请先运行 'conda install ffmpeg'")
            return

    print("所有视频抽帧完成！")

if __name__ == "__main__":
    extract_frames()