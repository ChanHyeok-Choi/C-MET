import os
import random
import numpy as np
import shutil

seed = 42
random.seed(seed)
np.random.seed(seed)


emotions = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']

data_root = '/home3/t20181257/EmotionalTFG/dataset/MEAD/FPS25'
test_IDs = ['M003', 'M030', 'W009', 'W015']

intensity = 'level_3'

emotions_audio_dict = {emotion: [] for emotion in emotions}

for speaker in os.listdir(data_root):
    if speaker not in test_IDs:
        continue
    speaker_path = os.path.join(data_root, speaker, 'front')
    for emotion in os.listdir(speaker_path):
        emotion_path = os.path.join(speaker_path, emotion, intensity)
        if emotion not in emotions:
            continue
        for audio_file in os.listdir(emotion_path):
            if audio_file.endswith('.wav'):
                emotions_audio_dict[emotion].append(os.path.join(emotion_path, audio_file))

num_samples = 20
for emotion, audio_files in emotions_audio_dict.items():
    sampled_files = random.sample(audio_files, num_samples)
    for sampled_file in sampled_files:
        _, _, _, _, _ , _, _, ID, _, emotion, intensity, filename = sampled_file.split('/')
        save_path = os.path.join('./', emotion, f'{ID}_{intensity}_{filename}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shutil.copy2(sampled_file, save_path)