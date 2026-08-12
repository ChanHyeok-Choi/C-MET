import cv2
import os
import argparse

def extract_frames(video_path, output_dir, prefix="frame", extension="jpg"):
    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Build the output file path (e.g. frame_0001.jpg).
        filename = os.path.join(output_dir, f"{prefix}_{frame_count:04d}.{extension}")
        cv2.imwrite(filename, frame)
        frame_count += 1

    cap.release()
    print(f"Saved {frame_count} frames to {output_dir}.")

if __name__ == '__main__':
    video_path = "/path/to/EmotionalTFG/dataset/MEAD/FPS25/M003/front/angry/level_3/003.mp4"
    output_dir = "./dataset1"
    extract_frames(video_path, output_dir)
