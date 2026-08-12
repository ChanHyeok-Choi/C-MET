# coding:utf-8
import os
import cv2
import glob
import shutil
import subprocess
import threading
import numpy as np
import pandas as pd
import argparse

# Script and model paths used for face alignment — these files are not
# included in this repo, so download them yourself into the location
# below (under the Emotion-FAN vendor directory)
FUNC_PATH = './Emotion-FAN/data/face_alignment_code/lib/face_align_cuda.py'
PREDICTOR_PATH = './Emotion-FAN/data/face_alignment_code/lib/shape_predictor_5_face_landmarks.dat'
CNN_FACE_DETECTOR = './Emotion-FAN/data/face_alignment_code/lib/mmod_human_face_detector.dat'

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def sample_frames(input_folder, output_folder, sample_count=16):
    """
    Sort the jpg files in input_folder, then use numpy's linspace to
    evenly sample sample_count of them into output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))
    if not files:
        print("No jpg files found in", input_folder)
        return False

    total = len(files)
    if total <= sample_count:
        sampled_files = files  # use them all if there are fewer than sample_count
    else:
        indices = np.linspace(0, total - 1, sample_count, dtype=int)
        sampled_files = [files[i] for i in indices]

    for f in sampled_files:
        shutil.copy2(f, output_folder)

    return True

def video2frame(video_path, output_dir, sample_count=16):
    """
    Extract frames from a video into memory, save them all to a temp
    folder, sample them via sample_frames, then call the face alignment
    script to crop faces.
    """
    # Save all frames of the video to a temp folder (output_dir/all_frames)
    all_frames_dir = os.path.join(output_dir, "all_frames")
    make_dir(all_frames_dir)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(all_frames_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1
    cap.release()

    if frame_idx == 0:
        print(f"No frames extracted from {video_path}")
        return

    # Temp folder for the sampled frames (output_dir/sampled_frames)
    sampled_dir = os.path.join(output_dir, "sampled_frames")
    success = sample_frames(all_frames_dir, sampled_dir, sample_count=sample_count)
    if not success:
        print("No frames sampled from", video_path)
        return

    # Run the face crop (face alignment) script
    linux_command = 'python {:} {:} "{:}" "{:}" {:} {:}'.format(
        FUNC_PATH, PREDICTOR_PATH, sampled_dir, output_dir, CNN_FACE_DETECTOR, 0
    )
    print(f'Processing video: {video_path}')
    status, output = subprocess.getstatusoutput(linux_command)
    print('Command status: {}\nOutput: {}'.format(status, output))

    # Remove the temp folders
    shutil.rmtree(all_frames_dir)
    shutil.rmtree(sampled_dir)

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    try:
        return any(filename.lower().endswith(ext) for ext in video_extensions)
    except Exception as e:
        # runs if an error occurs
        print("Error occurred:", e)
        print(filename)

class VideoThread(threading.Thread):
    def __init__(self, func, args):
        super(VideoThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.func(*self.args)

def run_threads(threads, n_thread):
    used_threads = []
    for num, new_thread in enumerate(threads):
        print('Starting thread index: {:}'.format(num))
        new_thread.start()
        used_threads.append(new_thread)

        if (num + 1) % n_thread == 0:
            for t in used_threads:
                t.join()
            used_threads = []

    # Wait for any remaining threads to finish.
    for t in used_threads:
        t.join()

def process_videos_from_df(df, video_path_col, frame_dir, n_thread=20, sample_count=16):
    threads = []
    for idx, row in df.iterrows():
        try:
            video_path = row[video_path_col]
            if not is_video_file(video_path):
                print(f"Row {idx}: '{video_path}' is not a video file.")
                continue

            # Create the output directory under frame_dir keyed by idx (e.g. 0000000, 0000001, ...)
            output_dir = os.path.join(frame_dir, f"{idx:07d}")
            df.at[idx, 'face_dir'] = output_dir
            make_dir(output_dir)

            # Queue a thread for frame extraction + face cropping.
            thread = VideoThread(video2frame, (video_path, output_dir, sample_count))
            threads.append(thread)
        except Exception as e:
            print(f"Error while processing row {idx}: {e}")

    run_threads(threads, n_thread)

# Example: build a DataFrame, then call process_videos_from_df
if __name__ == '__main__':
    # Read the example CSV file into a DataFrame.

    directory = "./runs/mead_ours"

    # Collect every CSV path.
    csv_paths = glob.glob(os.path.join(directory, "*.csv"))

    csv_list = []

    for csv_path in csv_paths:
        try:
            # Read only the header row.
            df = pd.read_csv(csv_path, nrows=0)
            columns = df.columns.tolist()
            # Get the name of the 5th column (index 4), or None if missing.
            col_name = columns[4] if len(columns) > 4 else None
            # col_name = "EAT"
            csv_list.append((csv_path, col_name))
        except Exception as e:
            csv_list.append((csv_path, f"Error: {e}"))
    for csv_info in csv_list:
        csv_path = csv_info[0]
        video_path_col = csv_info[1]
        csv_basename = os.path.basename(csv_path)  # filename only
        name_only = os.path.splitext(csv_basename)[0]
        df = pd.read_csv(csv_path)
        # Name of the column holding the video path (e.g. "video_path")

        # Root directory to store the frame/face results
        frame_dir = os.path.join("./runs/mead_ours/faces", name_only)
        # Remove the directory first if it already exists.
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        make_dir(frame_dir)

        process_videos_from_df(df, video_path_col, frame_dir, n_thread=8, sample_count=16)
        df.to_csv(csv_path, index=False)
