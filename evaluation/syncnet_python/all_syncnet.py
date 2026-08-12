import argparse
import glob
import os
import subprocess
from multiprocessing import Pool, cpu_count
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = "syncnet_python/workspace"
WORKER_COUNT = 8


def _run_syncnet(job: Dict[str, object]) -> Dict[str, object]:
    """Run run_syncnet.py for a single video, then read back the mean
    confidence it wrote to confidences/<reference>.txt (one line per
    detected face track) so the caller can store it in the CSV."""
    idx = job["idx"]
    video_path = job["video_path"]
    data_dir = job["data_dir"]
    reference = os.path.splitext(os.path.basename(video_path))[0]

    cmd = [
        "python",
        "run_syncnet.py",
        "--videofile",
        video_path,
        "--reference",
        reference,
        "--data_dir",
        data_dir,
    ]

    try:
        # run_syncnet.py lives in this same directory (syncnet_python/),
        # so pin cwd here rather than relying on the caller's cwd — this
        # script is meant to be launched as `python syncnet_python/all_syncnet.py`
        # from evaluation/, and a bare "run_syncnet.py" wouldn't resolve there.
        completed = subprocess.run(cmd, check=False, cwd=SCRIPT_DIR, capture_output=True)
        conf_txt_path = os.path.join(data_dir, "confidences", f"{reference}.txt")
        sync_conf = None
        if os.path.isfile(conf_txt_path):
            with open(conf_txt_path) as f:
                values = [float(line.strip()) for line in f if line.strip()]
            if values:
                sync_conf = sum(values) / len(values)
        return {
            "idx": idx,
            "video_path": video_path,
            "returncode": completed.returncode,
            "sync_conf": sync_conf,
            "error": None,
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "idx": idx,
            "video_path": video_path,
            "returncode": None,
            "sync_conf": None,
            "error": str(exc),
        }


def _extract_video_column(df: pd.DataFrame) -> str:
    generated_video = "gt"
    return generated_video if generated_video in df.columns else df.columns[4]


def _queue_jobs(df: pd.DataFrame, video_column: str, data_dir: str) -> (List[Dict[str, object]], List[str]):
    jobs: List[Dict[str, object]] = []
    video_errors: List[str] = []

    for idx, row in df.iterrows():
        if 'Sync_conf' in df.columns and pd.notna(row.get('Sync_conf')):
            continue
        video_path = row.get(video_column)
        if not video_path:
            continue

        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            video_errors.append(video_path)
            continue

        jobs.append({"idx": idx, "video_path": video_path, "data_dir": data_dir})

    return jobs, video_errors


def _process_csv(csv_path: str, pool: Pool) -> None:
    df = pd.read_csv(csv_path)
    if 'Sync_conf' not in df.columns:
        df['Sync_conf'] = pd.NA

    video_column = _extract_video_column(df)
    data_basename = os.path.splitext(os.path.basename(csv_path))[0]
    # Absolute path: run_syncnet.py runs with cwd=SCRIPT_DIR (syncnet_python/),
    # not this process's cwd, so a relative "syncnet_python/workspace/..."
    # string would resolve to the wrong (doubly-nested) location there.
    data_dir = os.path.abspath(os.path.join(WORKSPACE_ROOT, data_basename))

    try:
        jobs, video_errors = _queue_jobs(df, video_column, data_dir)
    except ValueError as exc:
        print(exc)
        return

    print(f"Starting {csv_path} ({len(jobs)} jobs)")
    run_failures = []
    save_interval = 50
    processed_since_save = 0

    if jobs:
        for result in tqdm(pool.imap_unordered(_run_syncnet, jobs), total=len(jobs), desc="Computing Sync_conf"):
            if result["error"] or result["returncode"]:
                run_failures.append(result)
            if result["sync_conf"] is not None:
                df.at[result["idx"], 'Sync_conf'] = result["sync_conf"]
                processed_since_save += 1
                if processed_since_save >= save_interval:
                    df.to_csv(csv_path, index=False)
                    processed_since_save = 0
                    tqdm.write(f"Checkpoint saved at row {result['idx']}")

    df.to_csv(csv_path, index=False)
    print(f"CSV updated with Sync_conf column: {csv_path}")

    print(f"video_errors: {video_errors}")

    if run_failures:
        print("Failed jobs:")
        for failure in run_failures:
            status = (
                f"returncode={failure['returncode']}"
                if failure["error"] is None
                else f"error={failure['error']}"
            )
            print(f"  {failure['video_path']} -> {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run run_syncnet.py for every row in a CSV and store the mean Sync_conf per row.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV to process.")
    parser.add_argument("--worker_count", type=int, default=WORKER_COUNT, help="Number of worker processes.")
    args = parser.parse_args()

    with Pool(processes=args.worker_count) as pool:
        _process_csv(args.csv_path, pool)


if __name__ == "__main__":
    main()
