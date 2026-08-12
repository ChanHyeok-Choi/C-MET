import glob
import os
import pandas as pd
import subprocess

directory = "/path/to/csv/new105"

# Collect every CSV path.
csv_paths = glob.glob(os.path.join(directory, "*.csv"))

csv_list = []

for csv_path in csv_paths:
    python_name = "emotion-fan.py"
    if "crema" in csv_path.lower():
        python_name = "emotion-fan_crema.py"
    elif "ravdess" in csv_path.lower():
        python_name = "emotion-fan_ravdess.py"
    cmd = [
        "python",
        python_name,
        "--csv_file", f"{csv_path}",
        "--num_frames", "16"
    ]

    # Run it.
    subprocess.run(cmd)
