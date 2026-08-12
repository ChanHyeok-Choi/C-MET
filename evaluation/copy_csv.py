import os
import shutil
import re

# Specify the txt file and destination directory paths
txt_path = "copy_list.txt"
parent_dir = "/path/to/csv/"
max_num = 0
for name in os.listdir(parent_dir):
    match = re.match(r'new(\d+)$', name)
    if match:
        num = int(match.group(1))
        if num > max_num:
            max_num = num

target_dir = f"/path/to/csv/new{max_num+1}"
# target_dir = f"/path/to/csv/new39"
# Create the destination directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

with open(txt_path, "r") as f:
    for line in f:
        src = line.strip()
        if src and os.path.isfile(src):  # only copy if the file actually exists
            filename = os.path.basename(src)
            dst = os.path.join(target_dir, filename)
            shutil.copy(src, dst)
            print(f"Copied: {src} -> {dst}")
        else:
            print(f"File not found or invalid line: {src}")

print("target dir : ", target_dir)
