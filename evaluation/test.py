# import pandas as pd

# # Read the CSV file (update the path to the actual file location)
# csv_path = "/path/to/SEVA/csv/test_EAT.csv"
# df = pd.read_csv(csv_path)

# # Prepend "hello" to each value in the "gt_video_path" column
# path = "/path/to/EAT_code/demo_MEAD/output/ser/"
# df["EAT+SER"] = path + df["EAT+SER"]

# # Save the result to a new CSV file (use the same filename to overwrite the original)
# df.to_csv(csv_path, index=False)
# import csv

# input_csv = '/path/to/SEVA/csv/MEAD/quantitative+EDTalk_250224.csv'   # existing CSV path
# output_csv = '/path/to/SEVA/csv/MEAD/inten3.csv' # path to save the new CSV to

# keywords = "level_3"
# label_to_emotion = {
#     'hap': 'happy',
#     'ang': 'angry',
#     'dis': 'disgusted',
#     'fea': 'fear',
#     'sad': 'sad',
#     'neu': 'neu',
#     'sur': 'surprised',
#     'con': 'contempt'
# }

# with open(input_csv, 'r', encoding='utf-8', newline='') as csv_in, \
#      open(output_csv, 'w', encoding='utf-8', newline='') as csv_out:
#     reader = csv.reader(csv_in)
#     writer = csv.writer(csv_out)

#     # Read the header row and append the new column name.
#     header = next(reader)
#     header.append('gt_video_path')  # name of the column to add
#     writer.writerow(header)

#     # Append a new value to each row and write it out.
#     for row in reader:
#         # Example: derive a new value from the existing row data, or add a constant.
#         file_path = row[2]
#         if keywords in file_path:
#             writer.writerow(row)

# print(f"New CSV file saved to {output_csv}")
# import os

# # Specify the parent directory path.
# parent_directory = "/path/to/SEVA/faces/RAVDESS_GT"

# # Loop from 1 to 1439 (numbers 1~1439).
# for i in range(1, 1440):
#     # Zero-pad the number to 7 digits (e.g. 1 -> 0000001).
#     dir_name = f"{i:07d}"
#     dir_path = os.path.join(parent_directory, dir_name)
#     # Check whether the directory exists.
#     if not os.path.isdir(dir_path):
#         print(f"Missing directory: {dir_name}")


import csv

input_csv = '/path/to/SEVA/csv/test_EAT.csv'   # existing CSV path
output_csv = '/path/to/SEVA/csv/test_EAT2.csv' # path to save the new CSV to

keywords = ['neutral', 'angry', 'happy', 'fear', 'disgusted', 'surprised', 'sad', 'contempt']
label_to_emotion = {
    'hap': 'happy',
    'ang': 'angry',
    'dis': 'disgusted',
    'fea': 'fear',
    'sad': 'sad',
    'neu': 'neu',
    'sur': 'surprised',
    'con': 'contempt'
}
names=[851,961,967,970,971,972,974,976,977,980,984,992,993,994,997,1000,1001,1003,1006,1008,1009,1014,1015]

with open(input_csv, 'r', encoding='utf-8', newline='') as csv_in, \
     open(output_csv, 'w', encoding='utf-8', newline='') as csv_out:
    reader = csv.reader(csv_in)
    writer = csv.writer(csv_out)

    # Read the header row and append the new column name.
    header = next(reader)
    header.append('gt_emotion')  # name of the column to add
    writer.writerow(header)

    # Append a new value to each row and write it out.
    for row in reader:
        # Example: derive a new value from the existing row data, or add a constant.
        file_path = row[0]
        matches = [kw for kw in keywords if kw in file_path][0]
        row.append(matches)
        writer.writerow(row)

print(f"New CSV file saved to {output_csv}")
# import pandas as pd

# # Specify the original CSV path and the path to save the new CSV to.
# input_csv = "/path/to/SEVA/csv/test_sameID_CE.csv"
# output_csv = "/path/to/SEVA/csv/test_sameID_CE2.csv"

# # Read the CSV file.
# df = pd.read_csv(input_csv)

# # Drop rows where gt_emotion is neutral (case-insensitive).
# filtered_df = df[~df["gt_emotion"].str.lower().eq("neutral")]

# # Save the result to a new CSV file (without the index).
# filtered_df.to_csv(output_csv, index=False)

# print(f"Filtered CSV file saved as {output_csv}")




# import csv, os
# from natsort import natsorted

# # Example list of video names.
# video_list1 = natsorted([f for f in os.listdir("/path/to/EAT_code/demo_MEAD/output/gt/") if f.endswith(".mp4")])
# video_list2 = natsorted([f for f in os.listdir("/path/to/EAT_code/demo_MEAD/output/ser/") if f.endswith(".mp4")])

# # Rows to write to the CSV file (each row is [v1, v2]).
# rows = []
# for v1, v2 in zip(video_list1, video_list2):
#     rows.append([v1, v2])

# output_csv = "test_EAT.csv"

# with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#     # Write the header.
#     writer.writerow(["EAT", "EAT+SER"])
#     # Write all rows at once.
#     writer.writerows(rows)

# print(f"CSV file saved to {output_csv}")


