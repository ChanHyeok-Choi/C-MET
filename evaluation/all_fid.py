import glob
import os
import pandas as pd
import subprocess
import sys

if len(sys.argv) < 2:
    print("Usage: python this_script.py <number>")
    exit(1)

num = sys.argv[1]   # takes a number (e.g. 53) as an argument
directory = f"/path/to/csv/new{num}"
frame_dir = "/path/to/SEVA/frames"

# Collect every CSV path.
csv_paths = glob.glob(os.path.join(directory, "*.csv"))

csv_list = []

for csv_path in csv_paths:
    filename = os.path.basename(csv_path)
    name_only = os.path.splitext(filename)[0]
    frame_path = os.path.join(frame_dir, name_only)
    cmd = [
        "python",
        "pytorch-fid/custom.py",
        f"{frame_path}",
        f"{frame_path}_GT",
    ]

    # Run it.
    subprocess.run(cmd)
