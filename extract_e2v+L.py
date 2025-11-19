'''
Using the emotion representation model
rec_result only contains {'feats'}
	granularity="utterance": {'feats': [*768]}
	granularity="frame": {feats: [T*768]}
'''
import os
from glob import glob
import torch
from funasr import AutoModel
import pandas as pd
import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--root', default='', type=str, help='path/to/audios')
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "iic/emotion2vec_plus_large"  # or "iic/emotion2vec_plus_large_zh" for Chinese version
model = AutoModel(
    model=model_id,
    hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
)

data_root = args.root
if data_root == './dataset/MEAD/FPS25':
    ID_list = os.listdir(data_root)
    ID_list.sort()
    for ID in ID_list:
        ID_path = os.path.join(data_root, ID, 'front')
        emotions = os.listdir(ID_path)
        emotions.sort()
        for emotion in emotions:
            emotion_path = os.path.join(ID_path, emotion)
            levels = os.listdir(emotion_path)
            levels.sort()
            for level in levels:
                level_path = os.path.join(emotion_path, level)
                wav_list = list(glob(os.path.join(level_path, '*.wav')))
                wav_list.sort()
                for wav in wav_list:
                    wav_name = os.path.basename(wav)
                    wav_name = wav_name.split('.')[0]
                    wav_path = os.path.join(level_path, wav_name + '.wav')
                    if not os.path.exists(wav_path):
                        print(f"File does not exist: {wav_path}")
                        continue
                    target_path = os.path.join(level_path, 'emotion2vec+large_features', wav_name + '.npy')
                    if not os.path.exists(target_path):
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        rec_result = model.generate(wav_path, output_dir=os.path.dirname(target_path), granularity="utterance")
else:
    for wav in os.listdir(data_root):
        if '.wav' not in wav:
            continue
        wav_path = os.path.join(data_root, wav)
        save_path = os.path.join(data_root, 'emotion2vec+large_features', wav.replace('.wav', '.npy'))
        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            rec_result = model.generate(wav_path, output_dir=os.path.dirname(save_path), granularity="utterance")


