# C-MET Evaluation Pipeline

Prepare a CSV (e.g. `dataset/MEAD/ours.csv`, not committed to this repo)
that references `dataset/MEAD/test.csv`, storing the path to each file
you generated in the 5th column, named `generated_path` (ground-truth
column: `gt_video_path`). Then:

```bash
mkdir -p evaluation/runs/mead_ours
cp dataset/MEAD/ours.csv evaluation/runs/mead_ours/
cd evaluation
```

## 1. Preprocessing

```bash
python vide2frame_custom.py    # frames for predicted + GT video → runs/mead_ours/frames/
python frame2face_custom.py    # face crops for predicted video → runs/mead_ours/faces/, writes face_dir back into ours.csv
```

## 2. FID

Uses the frame directories produced in step 1. Pass `--csv_path` so the
per-video FID gets written into the CSV's `FID` column (otherwise it's
console-output only and `check_quantitative_all.py` in step 6 won't see
it); this also enables skipping already-scored rows and periodic
checkpoint saving:
```bash
python pytorch-fid/custom.py runs/mead_ours/frames/ours runs/mead_ours/frames/ours_GT --csv_path runs/mead_ours/ours.csv
```

A single video's frame count is too small to use much GPU memory on its
own, so `InceptionV3` is loaded once and reused, and `--video_batch_size`
(default 16) groups that many videos' frames into one forward pass —
raise `--batch-size` (default 50) and/or `--video_batch_size` together to
actually make use of a large GPU:
```bash
python pytorch-fid/custom.py runs/mead_ours/frames/ours runs/mead_ours/frames/ours_GT --csv_path runs/mead_ours/ours.csv --batch-size 512 --video_batch_size 64
```
There's no way to force a fixed VRAM floor (PyTorch allocates on demand,
not to a target) — this is the actual lever: bigger grouped batches use
proportionally more memory.

## 3. FVD

TensorFlow 1's default session behavior is to pre-allocate most of the
GPU's memory up front regardless of actual need; this is now disabled
(`allow_growth = True`) so it only takes what it uses.
```bash
python fvd.py runs/mead_ours
```

## 4. Emotion accuracy (Accemo)

Checkpoints (Google Drive, not committed here — `evaluation/**/*.pth` is
already gitignored, so it's safe to download them straight into this tree):
- MEAD finetune: https://drive.google.com/file/d/1H0tqOEe5-EqlmomB_FujgbrG8C7dadf1/view?usp=sharing
  → save as `evaluation/Emotion-FAN/checkpoints/Emotion-FAN_MEAD.pth`
- CREMA-D finetune: https://drive.google.com/file/d/1QZQj39N05lx_3Qidhs3wzArx6SPdHXFU/view?usp=sharing
  → save as `evaluation/Emotion-FAN/checkpoints/Emotion-FAN_CREMA_D.pth`

Missing files under `Emotion-FAN/`: download from
https://github.com/Open-Debin/Emotion-FAN and place at the same relative path.

```bash
python Emotion-FAN/emotion-fan.py \
  --csv_file runs/mead_ours/ours.csv \
  --checkpoint Emotion-FAN/checkpoints/Emotion-FAN_MEAD.pth \
  --num_frames 16
```

The released checkpoints were finetuned with (historical record — the
`--csv_file` flag no longer exists on `emotion_finetune.py` in this repo,
which now takes `--train_csv`/`--test_csv` instead; adapt accordingly if
reproducing):
```bash
python emotion_finetune.py --batch_size 32 --lr 0.01 --num_frames 16 \
  --checkpoint_dir /path/to/SEVA/Emotion-FAN/pretrain_model/finetune \
  --csv_file /path/to/SEVA/train_dataset.csv
```

## 5. Sync confidence (Syncconf)

Missing files under `syncnet_python/`: download from
https://github.com/joonson/syncnet_python and place at the same relative path.

`all_pipeline.py` pairs the generated video (`generated_path`) with the
**source** audio (`source_audio_path`), not the ground-truth audio —
C-MET preserves the source's spoken content and only changes the emotion,
so the generated video's lips should sync to the source audio it was
driven by, not to `gt_video_path`'s own audio.

```bash
python syncnet_python/all_pipeline.py --csv_path runs/mead_ours/ours.csv
python syncnet_python/all_syncnet.py --csv_path runs/mead_ours/ours.csv
python syncnet_python/conf_mean.py --csv_path runs/mead_ours/ours.csv
```

## 6. Show all metrics

Required output: **FID, FVD, Sync_conf, Accemo**.

Steps 2, 3, 4, and `all_syncnet.py` in step 5 all write their results into
the CSV (`FID`, `fvd`, `predicted_emotion`, `Sync_conf` columns — step 2
only writes `FID` if you passed `--csv_path`; `Sync_conf` is the mean of
each video's face-track confidence values). Run this after steps 2-5:
```bash
python check_quantitative_all.py --csv_path runs/mead_ours/ours.csv
```

## 7. AITV timing methodology

Not computed by anything in this directory — it's measured inline in
`inference_dataset_ref.py` (the main C-MET inference script, outside
`evaluation/`) via its `infer_times` list. Pseudocode, read directly from
that file:

```
infer_times = []
for row in csv:
    t_start = now()

    # preprocessing IS included in the timed window:
    source_image = img_preprocessing(row.source_image)
    driving_audio_or_video = audio_preprocessing(...) or vid_preprocessing(...)

    for each output frame:
        frame = generate(source_image, driving_signal_for_this_frame)

    t_end = now()                      # stops here —
    infer_times.append(t_end - t_start)

    # NOT timed: ffmpeg mux/encode + save to disk
    save_video(frames, audio_path)

AITV = mean(infer_times)
```

Caveats, also read directly from the code, not assumed:
- No `torch.cuda.synchronize()` around the timed region — wall-clock delta
  may undercount async CUDA work.
- No explicit warm-up iteration — the first row's CUDA/cuDNN
  autotune overhead is included in the average.
- GPU model and precision aren't fixed in code (`--device` CLI flag,
  `weight_dtype` from a config file), so they aren't recoverable from the
  code alone.
