import os
import pickle

import cv2
import librosa
import numpy as np
from scipy import signal


class videoReader(object):
    def __init__(self, video_path, frame_interval=1, frame_kept_per_second=1):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.frame_kept_per_second = frame_kept_per_second

        # pdb.set_trace()
        self.vid = cv2.VideoCapture(self.video_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_len = int(self.video_frames / self.fps)

    def video2frame(self, frame_save_path):
        self.frame_save_path = frame_save_path
        success, image = self.vid.read()
        count = 0
        while success:
            count += 1
            if count % self.frame_interval == 0:
                save_name = "{}/frame_{}_{}.jpg".format(self.frame_save_path, int(count / self.fps), count)  # filename_second_index
                cv2.imencode(".jpg", image)[1].tofile(save_name)
            success, image = self.vid.read()

    def video2frame_update(self, frame_save_path, min_save_frame=3):
        self.frame_save_path = frame_save_path

        count = 0
        save_count = 0
        frame_interval = int(self.fps / self.frame_kept_per_second)
        while count < self.video_frames:
            ret, image = self.vid.read()
            if not ret:
                break
            if count % self.fps == 0:
                frame_id = 0
            if frame_id < frame_interval * self.frame_kept_per_second and frame_id % frame_interval == 0:
                save_name = "{0}/{1:05d}.jpg".format(self.frame_save_path, count)
                cv2.imencode(".jpg", image)[1].tofile(save_name)
                save_count += 1

            frame_id += 1
            count += 1

        if save_count < min_save_frame:
            add_count = min_save_frame - save_count
            count = 0
            if self.video_frames < min_save_frame:
                while count < add_count:
                    frame_id = np.random.randint(0, min_save_frame)
                    save_name = "{0}/{1:05d}.jpg".format(self.frame_save_path, frame_id)
                    if not os.path.exists(save_name):
                        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, image = self.vid.read()
                        cv2.imencode(".jpg", image)[1].tofile(save_name)
                        count += 1
            else:
                while count < add_count:
                    frame_id = np.random.randint(0, self.video_frames)
                    save_name = "{0}/{1:05d}.jpg".format(self.frame_save_path, frame_id)
                    if not os.path.exists(save_name):
                        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                        ret, image = self.vid.read()
                        cv2.imencode(".jpg", image)[1].tofile(save_name)
                        count += 1

    def video2frame_update_SE(self, frame_save_path, min_save_frame=3, start_t=0, end_t=10):
        self.frame_save_path = frame_save_path

        count = 0
        save_count = 0
        frame_interval = int(self.fps / self.frame_kept_per_second)
        
        # 用来记录实际保存了多少帧
        saved_frames_indices = []

        num_count = 0 # 对应秒数
        
        while count < self.video_frames:
            ret, image = self.vid.read()
            if not ret:
                break
            
            # 只有当正好落在 fps 间隔上时，才算作“一秒/一帧”
            if count % frame_interval == 0:
                # 检查是否在事件时间范围内
                if start_t <= num_count < end_t: # 注意这里建议用 < end_t
                    # ❗关键修改：如果已经存够了 min_save_frame (4帧)，就停止保存，直接跳出
                    if save_count >= min_save_frame:
                        break 
                    
                    save_name = "{0}/{1:05d}.jpg".format(self.frame_save_path, save_count) # 建议直接用 0,1,2,3 命名，方便读取
                    cv2.imencode(".jpg", image)[1].tofile(save_name)
                    save_count += 1
                
                num_count += 1 # 秒数增加

            count += 1

        # Padding 逻辑：如果没存够 4 帧（比如事件只有2秒），需要补帧
        if save_count < min_save_frame:
            add_count = min_save_frame - save_count
            # 简单的策略：复制最后一张图，或者随机采样事件内的图
            # 这里沿用你原本的随机采样逻辑，稍微简化
            while save_count < min_save_frame:
                # 随机从 start_t 到 end_t 之间找一秒
                rand_sec = np.random.randint(start_t, end_t) if end_t > start_t else start_t
                frame_pos = int(rand_sec * self.fps)
                
                self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, image = self.vid.read()
                
                if ret:
                    save_name = "{0}/{1:05d}.jpg".format(self.frame_save_path, save_count)
                    cv2.imencode(".jpg", image)[1].tofile(save_name)
                    save_count += 1
                else:
                    # 如果读不到，就跳出避免死循环（极少情况）
                    break


class AVE_dataset(object):
    def __init__(self, path_to_dataset=None, frame_interval=1, frame_kept_per_second=1):
        self.path_to_video = os.path.join(path_to_dataset, "AVE")
        self.path_to_audio = os.path.join(path_to_dataset, "Audios")
        self.frame_kept_per_second = frame_kept_per_second
        self.sr = 16000

        self.path_to_save = os.path.join(path_to_dataset, "Image-{:02d}-FPS-SE".format(self.frame_kept_per_second))
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)

        self.path_to_save_audio = os.path.join(path_to_dataset, "Audio-{:d}-SE".format(1004))
        if not os.path.exists(self.path_to_save_audio):
            os.mkdir(self.path_to_save_audio)

        # csv_file = pd.read_csv(os.path.join(path_to_dataset, 'Annotations.txt'))
        with open(os.path.join(path_to_dataset, "Annotations.txt"), "r") as f:
            self.file_list = f.readlines()

    def extractImage(self):

        for each_video in self.file_list[1:]:
            print("Processing {}".format(each_video), end="")
            each_video = each_video.split("&")
            video_dir = os.path.join(self.path_to_video, each_video[1] + ".mp4")
            self.videoReader = videoReader(video_path=video_dir, frame_kept_per_second=self.frame_kept_per_second)

            save_dir = os.path.join(self.path_to_save, each_video[1])
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.videoReader.video2frame_update(frame_save_path=save_dir, min_save_frame=10)  # 每个视频最少取10张图片

    def extractImage_SE(self):

        for each_video in self.file_list[1:]:
            print("Processing {}".format(each_video), end="")
            each_video = each_video.split("&")
            start_t = int(each_video[3])
            end_t = int(each_video[4])

            if end_t == 0:
                print("This ones is skipped")
                continue

            video_dir = os.path.join(self.path_to_video, each_video[1] + ".mp4")
            self.videoReader = videoReader(video_path=video_dir, frame_kept_per_second=self.frame_kept_per_second)

            save_dir = os.path.join(self.path_to_save, each_video[1])
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.videoReader.video2frame_update_SE(
                frame_save_path=save_dir,
                min_save_frame=4,
                start_t=start_t,
                end_t=end_t,
            )  # 每个视频最少取10张图片

    def extractWav(self):
        for each_audio in self.file_list[1:]:
            print("Processing {}".format(each_audio), end="")
            each_audio = each_audio.split("&")
            audio_dir = os.path.join(self.path_to_audio, each_audio[1] + ".wav")

            samples, rate = librosa.load(audio_dir, sr=self.sr)
            resamples = np.tile(samples, 10)[: self.sr * 10]
            resamples[resamples > 1.0] = 1.0
            resamples[resamples < -1.0] = -1.0

            # spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
            frequencies, times, spectrogram = signal.spectrogram(resamples, rate, nperseg=512, noverlap=353)
            spectrogram = np.log(np.abs(spectrogram) + 1e-7)
            mean = np.mean(spectrogram)
            std = np.std(spectrogram)
            spectrogram = np.divide(spectrogram - mean, std + 1e-9)
            # print(spectrogram.shape)
            save_name = os.path.join(self.path_to_save_audio, each_audio[1] + ".pkl")
            with open(save_name, "wb") as fid:
                pickle.dump(spectrogram, fid)

    def extractWav_SE(self):
        # --- 基于论文原文的参数配置 ---
        SR = 16000          # 原文: "resampled at 16 kHz"
        TARGET_SEC = 4      # 原文: "sampled at 1 fps for 4 frames" -> 4秒
        TARGET_LEN = SR * TARGET_SEC
        
        N_FFT = 512         # 原文: "window size of 512"
        HOP_LEN = 353       # 原文: "step size of 353"
        N_MELS = 128        # 论文未明确，但 Log-Mel 标准通常为 128 (也可能是 64/80，推荐先试 128)

        for each_audio in self.file_list[1:]:
            print(f"Processing {each_audio.strip()}")
            items = each_audio.split("&")
            video_name = items[1]
            try:
                start_t = float(items[3])
                end_t = float(items[4])
            except ValueError:
                continue

            audio_path = os.path.join(self.path_to_audio, video_name + ".wav")
            if not os.path.exists(audio_path):
                continue

            # 1. 裁剪 (Temporal Cropping)
            # 原文: "segments where the event occurs"
            duration = end_t - start_t
            if duration <= 0: continue
            
            # 加载指定区间
            y, _ = librosa.load(audio_path, sr=SR, offset=start_t, duration=duration)

            # 2. 统一长度到 4秒 (Padding / Truncating)
            if len(y) > TARGET_LEN:
                y = y[:TARGET_LEN] # 截断
            else:
                # 循环填充 (Tile) 
                # 原文提到 AVE 事件至少 2秒，因此 Tile 是安全的
                tile_count = int(np.ceil(TARGET_LEN / len(y)))
                y = np.tile(y, tile_count)[:TARGET_LEN]

            # 3. 生成 Log-Mel Spectrogram (关键修正)
            # 原文: "creating log-Mel spectrograms"
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=SR, 
                n_fft=N_FFT, 
                hop_length=HOP_LEN, 
                win_length=N_FFT,
                n_mels=N_MELS 
            )

            # 4. Log 变换
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)

            # 5. 标准化 (Optional but Recommended)
            # 你的旧代码里有做标准化，建议保留
            mean = np.mean(log_mel)
            std = np.std(log_mel)
            log_mel_norm = (log_mel - mean) / (std + 1e-7)

            # 保存
            save_name = os.path.join(self.path_to_save_audio, video_name + ".pkl")
            with open(save_name, "wb") as fid:
                pickle.dump(log_mel_norm, fid)


ave = AVE_dataset(path_to_dataset="/inspire/ssd/project/sais-scientific-models-evaluation/public/global_user/8331-huao/llj_temp/cxy/AVE_Dataset")
ave.extractImage_SE()
ave.extractWav_SE()
