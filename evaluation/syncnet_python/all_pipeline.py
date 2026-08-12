import argparse
import csv
import os
import subprocess
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = "syncnet_python/workspace"
AUDIO_COLUMN = "source_audio_path"
WORKER_COUNT = 8


def _run_syncnet(job: Dict[str, str]) -> Dict[str, object]:
    """Worker process that actually runs the SyncNet pipeline."""
    video_path = job["video_path"]
    audio_path = job["audio_path"]
    data_dir = job["data_dir"]
    reference = os.path.splitext(os.path.basename(video_path))[0]

    cmd = [
        "python",
        "run_pipeline.py",
        "--videofile",
        video_path,
        "--audiofile",
        audio_path,
        "--reference",
        reference,
        "--data_dir",
        data_dir,
    ]
    print("Running:", " ".join(cmd), flush=True)

    try:
        # run_pipeline.py lives in this same directory (syncnet_python/),
        # so pin cwd here rather than relying on the caller's cwd — this
        # script is meant to be launched as `python syncnet_python/all_pipeline.py`
        # from evaluation/, and a bare "run_pipeline.py" wouldn't resolve there.
        completed = subprocess.run(cmd, check=False, cwd=SCRIPT_DIR)
        return {
            "video_path": video_path,
            "audio_path": audio_path,
            "returncode": completed.returncode,
            "error": None,
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "video_path": video_path,
            "audio_path": audio_path,
            "returncode": None,
            "error": str(exc),
        }


def _extract_video_column(csv_path: str) -> str:
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

        generated_video = "gt"
        video_column = generated_video if generated_video in headers else headers[4]
        return video_column


def _queue_jobs(csv_path: str) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
    video_column = _extract_video_column(csv_path)
    data_basename = os.path.splitext(os.path.basename(csv_path))[0]
    # Absolute path: run_pipeline.py runs with cwd=SCRIPT_DIR (syncnet_python/),
    # not this process's cwd, so a relative "syncnet_python/workspace/..."
    # string would resolve to the wrong (doubly-nested) location there.
    data_dir = os.path.abspath(os.path.join(WORKSPACE_ROOT, data_basename))

    jobs: List[Dict[str, str]] = []
    video_errors: List[str] = []
    audio_errors: List[str] = []

    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_path = row.get(video_column)
            audio_raw = row.get(AUDIO_COLUMN)
            if video_path is None or audio_raw is None:
                continue

            audio_path = os.path.splitext(audio_raw)[0] + ".wav"

            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                video_errors.append(video_path)
                continue

            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                audio_errors.append(audio_path)
                continue

            jobs.append(
                {
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "data_dir": data_dir,
                }
            )

    return jobs, video_errors, audio_errors


def _process_csv(csv_path: str, pool: Pool) -> None:
    try:
        jobs, video_errors, audio_errors = _queue_jobs(csv_path)
    except ValueError as exc:
        print(exc)
        return

    print(f"Starting {csv_path} ({len(jobs)} jobs)")
    run_failures = []

    if jobs:
        for result in pool.map(_run_syncnet, jobs):
            if result["error"] or result["returncode"]:
                run_failures.append(result)

    print(f"video_errors: {video_errors}")
    print(f"audio_errors: {audio_errors}")

    if run_failures:
        print("Failed jobs:")
        for failure in run_failures:
            status = (
                f"returncode={failure['returncode']}"
                if failure["error"] is None
                else f"error={failure['error']}"
            )
            print(f"  {failure['video_path']} / {failure['audio_path']} -> {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SyncNet face-tracking pipeline for every row in a CSV.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV to process.")
    parser.add_argument("--worker_count", type=int, default=WORKER_COUNT, help="Number of worker processes.")
    args = parser.parse_args()

    with Pool(processes=args.worker_count) as pool:
        _process_csv(args.csv_path, pool)


if __name__ == "__main__":
    main()
