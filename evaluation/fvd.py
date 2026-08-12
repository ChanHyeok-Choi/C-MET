# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calculate FVD for video pairs from CSV files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import glob
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import frechet_video_distance as fvd
import cv2
from tqdm import tqdm
import argparse


def load_video(video_path, target_frames=15):
    """Load video from path and return as numpy array."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames loaded from {video_path}")

    # Resize or sample frames to target length
    if len(frames) > target_frames:
        # Sample frames uniformly
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    elif len(frames) < target_frames:
        # Repeat last frame
        while len(frames) < target_frames:
            frames.append(frames[-1])

    # Convert to numpy array and ensure shape is correct
    frames = np.array(frames)

    # Resize frames to 64x64 if necessary
    if frames.shape[1:3] != (64, 64):
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (64, 64))
            resized_frames.append(resized_frame)
        frames = np.array(resized_frames)

    return frames


def calculate_fvd_for_video_pair(video_path1, video_path2, sess,
                                  first_set_placeholder, second_set_placeholder,
                                  fvd_result):
    """Calculate FVD between two videos."""
    # Load videos - propagate exceptions to the caller on failure
    video1 = load_video(video_path1)
    video2 = load_video(video_path2)

    # Add batch dimension and replicate to make batch size 16
    video1_batch = np.tile(video1[np.newaxis, ...], (16, 1, 1, 1, 1))
    video2_batch = np.tile(video2[np.newaxis, ...], (16, 1, 1, 1, 1))

    # Calculate FVD
    fvd_value = sess.run(fvd_result, feed_dict={
        first_set_placeholder: video1_batch,
        second_set_placeholder: video2_batch
    })

    return fvd_value


def process_csv_directory(directory_path, output_directory=None):
    """Process all CSV files in the directory and add FVD column."""

    if output_directory is None:
        output_directory = directory_path

    os.makedirs(output_directory, exist_ok=True)

    # Find all CSV files
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return

    print(f"\nFound {len(csv_files)} CSV files in {directory_path}")

    # Build TensorFlow graph
    with tf.Graph().as_default():
        # Placeholders for video pairs (same configuration as the reference code)
        # NUMBER_OF_VIDEOS = 16, VIDEO_LENGTH = 15
        first_set_placeholder = tf.placeholder(
            tf.float32, [16, 15, 64, 64, 3])
        second_set_placeholder = tf.placeholder(
            tf.float32, [16, 15, 64, 64, 3])

        # Calculate FVD (same approach as the reference code)
        print("Initializing FVD model...")
        fvd_result = fvd.calculate_fvd(
            fvd.create_id3_embedding(fvd.preprocess(first_set_placeholder, (224, 224))),
            fvd.create_id3_embedding(fvd.preprocess(second_set_placeholder, (224, 224)))
        )

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            successful = 0
            failed = 0

            # Process each CSV file with tqdm
            for csv_file in tqdm(csv_files, desc="Processing CSV files", ncols=80):
                try:
                    tqdm.write(f"\n{'='*80}")
                    tqdm.write(f"Processing: {os.path.basename(csv_file)}")
                    tqdm.write(f"{'='*80}")

                    # Read CSV using pandas to check columns
                    df = pd.read_csv(csv_file)

                    if len(df) == 0:
                        raise ValueError(f"Empty CSV file: {csv_file}")

                    # Check if 'gt' column exists
                    generated_video = "EAT"
                    has_gt_column = generated_video in df.columns

                    if has_gt_column:
                        tqdm.write(f"  Using '{generated_video}' column for video path")
                        gt_col_name = generated_video
                    else:
                        # Use 5th column (index 4)
                        if len(df.columns) < 5:
                            raise ValueError(f"CSV file has less than 5 columns")
                        gt_col_name = df.columns[4]
                        tqdm.write(f"  Using column '{gt_col_name}' (5th column) for video path")

                    # Check for gt_video_path column (index 1)
                    if 'gt_video_path' not in df.columns:
                        raise ValueError("'gt_video_path' column not found")

                    # Add FVD column if not exists
                    if 'fvd' not in df.columns:
                        df['fvd'] = None

                    # Process each row with tqdm
                    fvd_values = []
                    failed_rows = 0
                    save_interval = 50
                    processed_since_save = 0

                    for idx in tqdm(df.index, desc="Processing videos", ncols=80, leave=False):
                        # Skip rows that already have an FVD value.
                        if pd.notna(df.loc[idx, 'fvd']):
                            fvd_values.append(df.loc[idx, 'fvd'])
                            continue

                        # Get video paths
                        video_path1 = str(df.loc[idx, 'gt_video_path']).strip('"')
                        video_path2 = str(df.loc[idx, gt_col_name]).strip('"')

                        try:
                            # Calculate FVD
                            fvd_value = calculate_fvd_for_video_pair(
                                video_path1, video_path2, sess,
                                first_set_placeholder, second_set_placeholder,
                                fvd_result
                            )

                            df.loc[idx, 'fvd'] = fvd_value
                            fvd_values.append(fvd_value)

                        except Exception as e:
                            tqdm.write(f"❌ Row {idx+1}: {str(e)[:100]}")
                            df.loc[idx, 'fvd'] = None
                            failed_rows += 1

                        processed_since_save += 1
                        if processed_since_save >= save_interval:
                            df.to_csv(csv_file, index=False)
                            processed_since_save = 0
                            tqdm.write(f"Checkpoint saved at row {idx}")

                    # Save to original CSV file
                    df.to_csv(csv_file, index=False)

                    # Print stats
                    valid_fvds = [f for f in fvd_values if f is not None and isinstance(f, (int, float))]
                    tqdm.write(f"\n✓ Completed: {os.path.basename(csv_file)}")
                    tqdm.write(f"  Total: {len(df)} | Success: {len(valid_fvds)} | Failed: {failed_rows}")
                    if valid_fvds:
                        tqdm.write(f"  Mean: {sum(valid_fvds)/len(valid_fvds):.4f} | Min: {min(valid_fvds):.4f} | Max: {max(valid_fvds):.4f}")

                    successful += 1

                except Exception as e:
                    tqdm.write(f"\n❌ Error processing {os.path.basename(csv_file)}:")
                    tqdm.write(f"  {str(e)}")
                    failed += 1

            # Final summary
            print(f"\n{'='*80}")
            print(f"COMPLETED")
            print(f"{'='*80}")
            print(f"Total CSV files: {len(csv_files)}")
            print(f"Successful: {successful} | Failed: {failed}")
            print(f"{'='*80}")


def main(argv):
    del argv

    parser = argparse.ArgumentParser(
        description='Calculate FVD for video pairs in CSV files')
    parser.add_argument('directory', type=str,
                        help='Directory containing CSV files')
    parser.add_argument('--output_directory', type=str, default=None,
                        help='Output directory (default: overwrites original files)')

    args = parser.parse_args()

    process_csv_directory(args.directory, args.output_directory)


if __name__ == "__main__":
    tf.app.run(main)