import subprocess
import os
import threading
import pandas as pd
import glob

VIDEO_EXTENSIONS = ['mp4', 'webm', 'avi']

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def video2frame(video_input, frame_output):
    # Build an ffmpeg command to dump the video as frame images.
    linux_cmd = 'ffmpeg -i "{:}" -f image2 "{:}/%07d.jpg"'.format(video_input, frame_output)
    print('Processing video: {}'.format(video_input))
    subprocess.getstatusoutput(linux_cmd)

class VideoThread(threading.Thread):
    def __init__(self, func, args):
        super(VideoThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.func(*self.args)

def run_threads(threads, n_thread):
    used_threads = []
    for num, thread in enumerate(threads):
        print('Starting thread index: {}'.format(num))
        thread.start()
        used_threads.append(thread)

        # Wait for the running threads every n_thread iterations.
        if (num + 1) % n_thread == 0:
            for t in used_threads:
                t.join()
            used_threads = []

    # Wait for any remaining threads to finish.
    for t in used_threads:
        t.join()

def process_videos_from_df(df, video_path_col, frame_dir, n_thread=20):
    threads = []
    for idx, row in df.iterrows():
        try:
            video_path = row[video_path_col]
            if not is_video_file(video_path):
                print(f"Row {idx}: '{video_path}' is not a video file.")
                continue

            # Create the output directory under frame_dir keyed by idx (e.g. 0000000, 0000001, ...).
            output_dir = os.path.join(frame_dir, f"{idx:07d}")
            make_dir(output_dir)

            # Queue a thread for the video -> frames conversion.
            thread = VideoThread(video2frame, (video_path, output_dir))
            threads.append(thread)
        except Exception as e:
            print(f"Error while processing row {idx}: {e}")

    run_threads(threads, n_thread)

if __name__ == '__main__':
    directory = "./runs/mead_ours"

    # Collect every CSV path.
    csv_paths = glob.glob(os.path.join(directory, "*.csv"))

    csv_list = []

    for csv_path in csv_paths:
        csv_basename = os.path.basename(csv_path)  # filename only
        name_only = os.path.splitext(csv_basename)[0]
        df = pd.read_csv(csv_path)  # CSV containing the video paths
        column_name = df.columns[4]
        # column_name = "EAT"
        frame_output_dir = os.path.join("./runs/mead_ours/frames", name_only)
        process_videos_from_df(df, column_name, frame_output_dir, n_thread=4)
        gt_dir = os.path.join("./runs/mead_ours/frames", name_only + '_GT')
        column_name = 'gt_video_path'
        process_videos_from_df(df, column_name, gt_dir, n_thread=4)
