import os
from glob import glob
from os.path import join, isfile

import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

import numpy as np
import pickle


class Dataset(TorchDataset):
    def __init__(self, split, dataset_root='./dataset/MEAD/FPS25', T=50, mode='mean', 
                num_feats=10, direction='average', num_samples=10, except_emotions=None, 
                audio_encoder='emotion2vec+large', scale_intensity=False, ID='same',
                feature_type='ED'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ID_lists = self.get_ID_list(split)  # ['M003', ...]
        self.dataset_root = dataset_root
        # self.T = 2*T
        self.T = T
        self.mode = mode
        self.num_feats = num_feats
        self.direction = direction
        self.except_emotions = except_emotions
        self.scale_intensity = scale_intensity
        self.ID = ID
        if feature_type == 'ED':
            self.feature_type = '_ED_exp.npy'
        elif feature_type == 'PD_FGC':
            self.feature_type = '_PD_FGC_emo.npy'
        else:
            raise ValueError("Invalid feature type, must be 'ED' or 'PD_FGC'")
        print("T:", self.T)
        print("Data will be processed in mode:", self.mode, ", direction:", self.direction)
        print("except_emotions:", self.except_emotions)
        print("scale_intensity:", self.scale_intensity)
        print(f"using {self.ID} ID expressions")

        self.all_video_names = []

        self.num_samples = num_samples
        if 'MEAD' in dataset_root:
            self.emotion_label = {'angry': 0, 'contempt': 1, 'disgusted': 2, 'fear': 3, 
                                    'happy': 4, 'neutral': 5, 'sad': 6, 'surprised': 7}
            self.idx2emotion = {v: k for k, v in self.emotion_label.items()}
            self.intensity = ['level_1', 'level_2', 'level_3']
            self.e2v_paths = {emotion: {intensity: [] for intensity in self.intensity} for emotion in self.emotion_label.keys()}
            self.init_e2v_MEAD_paths(audio_encoder)
            self.init_MEAD_video()

    def init_MEAD_video(self):
        count = 0
        print("loading video paths...")
        for ID in tqdm(self.ID_lists, total=len(self.ID_lists)):
            video_paths = list(glob(join(self.dataset_root, ID, '*/*/*/*.mp4')))
            video_paths.sort()
            filtered_paths = []
            for path in video_paths:
                parts = path.split(os.sep)
                emotion = parts[-3]
                filename = parts[-1]
                num = int(filename.split('.')[0])
                if emotion.lower() == "neutral":
                    if 1 <= num <= 3 or 31 <= num <= 40:
                        filtered_paths.append(path)
                    else:
                        count += 1
                else:
                    if 1 <= num <= 3 or 21 <= num <= 30:
                        filtered_paths.append(path)
                    else:
                        count += 1
            self.all_video_names.extend(filtered_paths)
        print("filtered out: ", count)
        print("complete, with available vids: ", len(self.all_video_names))
        print("Samples: ", self.all_video_names[-15:])
        # 감정별 샘플 비율 출력
        emotion_count = {k: 0 for k in self.emotion_label.keys()}
        for video_path in self.all_video_names:
            _, _, _, _, _, _, emotion, _, _ = video_path.split('/')
            emotion_count[emotion] += 1
        print("Emotion count: ", emotion_count)

    def init_e2v_MEAD_paths(self, audio_encoder):
        for ID in self.ID_lists:
            for emotion in self.emotion_label.keys():
                for intensity in self.intensity:
                    if emotion == 'neutral':
                        if intensity != 'level_1':
                            continue
                        indices = list(range(1, 41))
                    else:
                        indices = list(range(1, 31))
                    
                    for idx in indices:
                        path = f"dataset/MEAD/FPS25/{ID}/front/{emotion}/{intensity}/{audio_encoder}_features/{str(idx).zfill(3)}.npy"
                        if os.path.exists(path):
                            self.e2v_paths[emotion][intensity].append(path)
        print("e2v paths initialized.")
        print("e2v paths count: ", {emotion: {intensity: len(paths) for intensity, paths in self.e2v_paths[emotion].items()} for emotion in self.emotion_label.keys()})


    def get_e2v(self, emotion_1, emotion_2, intensity, index=None):
        # e2v path lists
        if 'MEAD' in self.dataset_root:
            emo_1_paths = self.e2v_paths[emotion_1][intensity] if emotion_1 != 'neutral' else self.e2v_paths[emotion_1]['level_1']
            emo_2_paths = self.e2v_paths[emotion_2][intensity] if emotion_2 != 'neutral' else self.e2v_paths[emotion_2]['level_1']
            strength = float(intensity.split('_')[-1])


        # deterministic index slicing
        if index is None:
            index = 0  # fallback if no index provided
        total_emo_1 = len(emo_1_paths)
        total_emo_2 = len(emo_2_paths)

        # offset을 두어 index가 길이를 넘지 않도록 함
        emo_1_indices = [(index + i) % total_emo_1 for i in range(self.num_samples)]
        emo_2_indices = [(index + i * 7) % total_emo_2 for i in range(self.num_samples)]  # 7은 skip step (optional)

        emo_1 = torch.stack([torch.from_numpy(np.load(emo_1_paths[i])).float() for i in emo_1_indices])
        emo_2 = torch.stack([torch.from_numpy(np.load(emo_2_paths[i])).float() for i in emo_2_indices])

        # normalize and subtract
        emo_1 = F.normalize(emo_1, p=2, dim=1)
        emo_2 = F.normalize(emo_2, p=2, dim=1)
        emo_1 = emo_1.mean(dim=0)
        emo_2 = emo_2.mean(dim=0)
        e2v = emo_2 - emo_1
        e2v = F.normalize(e2v, p=2, dim=0)

        e2v = e2v * strength

        return e2v, emo_1, emo_2

    def get_raw_e2v(self, emotion_1, emotion_2, intensity, index=None):
        # e2v path lists
        if 'MEAD' in self.dataset_root:
            emo_1_paths = self.e2v_paths[emotion_1][intensity] if emotion_1 != 'neutral' else self.e2v_paths[emotion_1]['level_1']
            emo_2_paths = self.e2v_paths[emotion_2][intensity] if emotion_2 != 'neutral' else self.e2v_paths[emotion_2]['level_1']

        # deterministic index slicing
        if index is None:
            index = 0  # fallback if no index provided
        total_emo_1 = len(emo_1_paths)
        total_emo_2 = len(emo_2_paths)

        # offset을 두어 index가 길이를 넘지 않도록 함
        emo_1_indices = [(index + i) % total_emo_1 for i in range(self.num_samples)]
        emo_2_indices = [(index + i * 7) % total_emo_2 for i in range(self.num_samples)]  # 7은 skip step (optional)

        emo_1 = torch.stack([torch.from_numpy(np.load(emo_1_paths[i])).float() for i in emo_1_indices])
        emo_2 = torch.stack([torch.from_numpy(np.load(emo_2_paths[i])).float() for i in emo_2_indices])

        emo_1 = emo_1.mean(dim=0)
        emo_2 = emo_2.mean(dim=0)
        e2v = emo_2 - emo_1

        return e2v, emo_1, emo_2

    def get_ID_list(self, split, dataset_name="MEAD"):
        vid_name_list = []
        with open(f'./dataset/{dataset_name}/{split}.txt') as f:
            for line in f:
                line = line.strip()
                if ' ' in line:
                    line = line.split()[0]
                vid_name_list.append(line)
        return vid_name_list
    
    def __len__(self):
        return len(self.all_video_names)

    def __getitem__(self, idx, target_id=None, target_emotion_1=None, target_emotion_2=None, target_intensity=None, target_num=None):
        max_attempts = 1000
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            vid_idx = random.randint(0, len(self.all_video_names) - 1)
            video_path = self.all_video_names[vid_idx]
            parts = video_path.split('/')
            _, _, _, _, ID, _, emotion_1, intensity, _ = parts
            emotion_2 = emotion_1
            while emotion_2 == emotion_1:
                emotion_2 = random.choice(list(self.emotion_label.keys()))
            if emotion_1 != 'neutral':
                emo_num = parts[-1].replace(".mp4", "")
                if int(emo_num) <= 3:
                    neu_num = emo_num
                else:
                    neu_num = str(int(emo_num) + 10).zfill(3)
            else:
                emo_num = parts[-1].replace(".mp4", "")
                neu_num = emo_num
            if self.ID == 'same':
                ID = ID
            elif self.ID == 'diff':
                ID = random.choice(self.ID_lists)
            else:
                raise ValueError("Invalid ID type, must be 'same' or 'diff'")
            if target_id is not None:
                ID = target_id
            if target_emotion_1 is not None:
                emotion_1 = target_emotion_1
            if target_emotion_2 is not None:
                emotion_2 = target_emotion_2
            if target_intensity is not None:
                intensity = target_intensity
            if target_num is not None:
                neu_num = target_num
                emo_num = target_num
                if int(emo_num) <= 3:
                    neu_num = emo_num
                else:
                    neu_num = str(int(emo_num) + 10).zfill(3)

            if self.except_emotions is not None and emotion_1 in self.except_emotions and emotion_2 in self.except_emotions:
                continue

            # e2v 데이터 샘플링 (10개 평균)
            if self.scale_intensity:
                e2v, e2v_emo_1, e2v_emo_2 = self.get_e2v(emotion_1, emotion_2, intensity, index=idx)
            else:
                e2v, e2v_emo_1, e2v_emo_2 = self.get_raw_e2v(emotion_1, emotion_2, intensity, index=idx)

            # opt 01. 같은 감정의 10개 데이터 평균 사용, ID를 random하게 선택
            if self.mode == 'mean':
                sampled_ED_emo_1, sampled_ED_emo_2 = [], []
                while len(sampled_ED_emo_1) < self.num_feats:
                    # random_id = random.choice(self.ID_lists)
                    if emotion_1 != 'neutral':
                        random_emo_1_num = random.choice([str(i).zfill(3) for i in list(range(1, 11)) + list(range(21, 31))])
                        emo_1_video_path = join(self.dataset_root, ID, 'front', emotion_1, intensity, random_emo_1_num + self.feature_type)
                    else:
                        random_emo_1_num = random.choice([str(i).zfill(3) for i in list(range(1, 11)) + list(range(31, 41))])
                        emo_1_video_path = join(self.dataset_root, ID, 'front', emotion_1, 'level_1', random_emo_1_num + self.feature_type)
                    if emotion_2 != 'neutral':
                        random_emo_2_num = random.choice([str(i).zfill(3) for i in list(range(1, 11)) + list(range(21, 31))])
                        emo_2_video_path = join(self.dataset_root, ID, 'front', emotion_2, intensity, random_emo_2_num + self.feature_type)
                    else:
                        random_emo_2_num = random.choice([str(i).zfill(3) for i in list(range(1, 11)) + list(range(31, 41))])
                        emo_2_video_path = join(self.dataset_root, ID, 'front', emotion_2, 'level_1', random_emo_2_num + self.feature_type)
                    if not isfile(emo_1_video_path) or not isfile(emo_2_video_path):
                        print("File not found:", emo_1_video_path, emo_2_video_path)
                        continue
                    emo_1_video = np.load(emo_1_video_path)
                    emo_2_video = np.load(emo_2_video_path)
                    emo_1_video = torch.from_numpy(emo_1_video).float()
                    emo_2_video = torch.from_numpy(emo_2_video).float()
                    sampled_ED_emo_1.append(emo_1_video)
                    sampled_ED_emo_2.append(emo_2_video)

                # 시간 길이 맞추기 (가장 긴 영상에 맞춰 반복)
                T_emo_1 = max([video.size(0) for video in sampled_ED_emo_1])
                for i in range(len(sampled_ED_emo_1)):
                    video = sampled_ED_emo_1[i]
                    if video.size(0) < T_emo_1:
                        repeat_times = T_emo_1 // video.size(0) + 1
                        video = torch.cat([video] * repeat_times, dim=0)[:T_emo_1, :]
                        sampled_ED_emo_1[i] = video
                T_emo_2 = max([video.size(0) for video in sampled_ED_emo_2])
                for i in range(len(sampled_ED_emo_2)):
                    video = sampled_ED_emo_2[i]
                    if video.size(0) < T_emo_2:
                        repeat_times = T_emo_2 // video.size(0) + 1
                        video = torch.cat([video] * repeat_times, dim=0)[:T_emo_2, :]
                        sampled_ED_emo_2[i] = video
                ED_emo_1 = torch.stack(sampled_ED_emo_1).mean(dim=0)
                ED_emo_2 = torch.stack(sampled_ED_emo_2).mean(dim=0)
                
                T_len = min(ED_emo_1.size(0), ED_emo_2.size(0))
                ED_emo_1 = ED_emo_1[:T_len, :]
                ED_emo_2 = ED_emo_2[:T_len, :]
                emo_dir = ED_emo_2 - ED_emo_1
            elif self.mode == 'single':
                # opt 02. 감정 1개 데이터 사용
                if emotion_1 != 'neutral':
                    emo_1_video_path = join(self.dataset_root, ID, 'front', emotion_1, intensity, emo_num + self.feature_type)
                else:
                    emo_1_video_path = join(self.dataset_root, ID, 'front', emotion_1, 'level_1', neu_num + self.feature_type)
                if emotion_2 != 'neutral':
                    emo_2_video_path = join(self.dataset_root, ID, 'front', emotion_2, intensity, emo_num + self.feature_type)
                else:
                    emo_2_video_path = join(self.dataset_root, ID, 'front', emotion_2, 'level_1', neu_num + self.feature_type)
                if not isfile(emo_1_video_path) or not isfile(emo_2_video_path):
                    print("File not found:", emo_1_video_path, emo_2_video_path)
                    continue
                ED_emo_1 = torch.from_numpy(np.load(emo_1_video_path)).float()
                ED_emo_2 = torch.from_numpy(np.load(emo_2_video_path)).float()

                emo_dir = ED_emo_2 - ED_emo_1

                T_len = emo_dir.size(0)
            else:
                raise ValueError("Invalid mode. Choose 'mean' or 'single'.")

            if self.direction == 'first':
                emo_dir = emo_dir[0:1, :].repeat(emo_dir.size(0), 1)
            elif self.direction == 'max':
                max_idx = emo_dir.mean(dim=1).argmax()
                emo_dir = emo_dir[max_idx].unsqueeze(0).repeat(emo_dir.size(0), 1)
            elif self.direction == 'average':
                emo_dir = torch.mean(emo_dir, dim=0, keepdim=True).repeat(emo_dir.size(0), 1)
            elif self.direction == 'random':
                random_ED_emo_1 = random.choice(ED_emo_1)
                random_ED_emo_2 = random.choice(ED_emo_2)
                emo_dir = random_ED_emo_2 - random_ED_emo_1
                emo_dir = emo_dir.unsqueeze(0).repeat(ED_emo_2.size(0), 1)
            elif self.direction == 'raw':
                emo_dir = ED_emo_2 - ED_emo_1

            if T_len < self.T:
                continue

            start_frame = random.randint(0, T_len - self.T)
            end_frame = start_frame + self.T
            emo_dir_ = emo_dir[start_frame:end_frame, :]

            ED_emo_1 = ED_emo_1[start_frame:end_frame, :]
            ED_emo_2 = ED_emo_2[start_frame:end_frame, :]
            
            # ED_ref는 emo_dir의 직전 self.T 만큼의 프레임을 사용
            if start_frame >= self.T:
                ED_ref = emo_dir[start_frame - self.T:start_frame, :]
            else:
                # 만약 start_frame의 위치가 self.T보다 작으면 ED_ref는 모자란만큼 0으로 패딩
                ED_ref = torch.zeros(self.T, emo_dir.size(1))
                if start_frame > 0:
                    ED_ref[-start_frame:, :] = emo_dir[:start_frame, :]

            # angry --> torch.tensor(0), contempt --> torch.tensor(1), ...
            emotion_label = self.emotion_label[emotion_2]
            emotion_label = torch.tensor(emotion_label).long()
            emotion_label = emotion_label.unsqueeze(0).repeat(ED_ref.size(0), 1)

            return e2v.unsqueeze(0), ED_ref, ED_emo_1, ED_emo_2, emo_dir_, emotion_label, e2v_emo_2.unsqueeze(0), e2v_emo_1.unsqueeze(0)

        # 만약 max_attempts까지 valid sample을 찾지 못하면 IndexError 발생
        raise IndexError("Valid sample not found after several attempts.")
        
if __name__ == '__main__':
    dataset = Dataset('train', dataset_root='./dataset/MEAD/FPS25', T=5, mode='mean', 
                    direction='average', num_feats=5, num_samples=10, except_emotions=['neutral'],
                    ID='diff', feature_type='PD_FGC', audio_encoder='emotion2vec+large')
    e2v, ED_ref, ED_emo_1, ED_emo_2, mean_emo_dir, el, e2v_emo_2, e2v_emo_1 = dataset[42]
    print(e2v.shape, ED_ref.shape, ED_emo_1.shape, ED_emo_2.shape, mean_emo_dir.shape, el.shape, e2v_emo_2.shape, e2v_emo_1.shape)
    # dataset = Dataset('test', dataset_root='./dataset/RAVDESS/FPS25', T=5, mode='mean', direction='average', num_samples=10, except_emotions=['neutral'])
    # e2v = dataset.get_e2v('angry', '02')
    # print(e2v.shape)
    # dataset = Dataset('test', dataset_root='./dataset/CREMA_D/FPS25', T=5, mode='mean', direction='average', num_samples=10, except_emotions=['neutral'])
    # e2v, emo, neu = dataset.get_e2v('ANG', 'MD')
    # print(e2v.shape)