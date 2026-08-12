import glob
import os
import pandas as pd
import subprocess

directory = "/path/to/csv/new72"

# Collect every CSV path.
csv_paths = glob.glob(os.path.join(directory, "*.csv"))

csv_list = []

for csv_path in csv_paths:
    cmd = [
        "python",
        "emotion-fan.py",
        "--csv_file", f"{csv_path}",
        "--checkpoint", "/path/to/SEVA/Emotion-FAN/checkpoints/checkpoint_epoch_12_baseline.pth",
        "--num_frames", "16"
    ]

    # Run it.
    subprocess.run(cmd)
