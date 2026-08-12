import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the mean SyncNet confidence for a CSV's clips.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV that was run through all_pipeline.py / all_syncnet.py.")
    args = parser.parse_args()

    csv_basename = os.path.basename(args.csv_path)  # e.g. "file.csv"
    csv_name = os.path.splitext(csv_basename)[0]  # e.g. "file"

    txt_dir = f'syncnet_python/workspace/{csv_name}/confidences'
    print(txt_dir)

    values = []

    for fname in os.listdir(txt_dir):
        if fname.endswith('.txt'):
            path = os.path.join(txt_dir, fname)
            with open(path) as f:
                try:
                    value = float(f.read().strip())
                    values.append(value)
                except ValueError:
                    print(f"Could not read a number from file {fname}")

    if values:
        mean_value = sum(values) / len(values)
        print(f"{csv_name} mean value: {mean_value}")
    else:
        print("No values were read.")
