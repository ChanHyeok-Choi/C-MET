import os
import csv
import argparse

def main(train_txt_path, base_dataset_path, output_csv):
    # Read the IDs from train.txt.
    with open(train_txt_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]

    # Define the emotion and level lists.
    emotions = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    levels = ['level_1', 'level_2', 'level_3']

    rows = []
    idx = 0
    face_dir = "/path/to/SEVA/faces/emotion_finetune/"
    # Build the directory for each id/emotion/level combination and find its mp4 files.
    for id_val in ids:
        for emotion in emotions:
            for level in levels:
                # Path layout: /path/to/EmotionalTFG/dataset/MEAD/FPS25/<id>/front/<emotion>/<level>/
                dir_path = os.path.join(base_dataset_path, id_val, 'front', emotion, level)
                if not os.path.isdir(dir_path):
                    continue
                # Process every mp4 file in that directory.
                for filename in os.listdir(dir_path):
                    if filename.lower().endswith('.mp4'):
                        video_path = os.path.join(dir_path, filename)
                        rows.append([emotion, video_path, f"{face_dir}{idx:07d}"])
                        idx += 1

    # Save as a CSV file (header: emotion, video_path).
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['emotion', 'video_path', 'face_dir'])
        writer.writerows(rows)

    print(f"CSV file saved to: {output_csv}")

if __name__ == '__main__':
    train_txt = "/path/to/SEVA/Emotion-FAN/train.txt"
    base_path = "/path/to/EmotionalTFG/dataset/MEAD/FPS25"
    output_csv = "./train_dataset.csv"
    main(train_txt, base_path, output_csv)
