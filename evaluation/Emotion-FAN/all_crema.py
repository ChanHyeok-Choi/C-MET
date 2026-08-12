import glob
import os
import pandas as pd
import subprocess

directory = "/path/to/SEVA/Emotion-FAN/csv/crema/test"

# Collect every CSV path.
csv_paths = glob.glob(os.path.join(directory, "*.csv"))

csv_list = []

for csv_path in csv_paths:
    cmd = [
        "python",
        "emotion-fan_crema.py",
        "--csv_file", f"{csv_path}",
        "--num_frames", "16"
    ]

    # Run it.
    subprocess.run(cmd)
