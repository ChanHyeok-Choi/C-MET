import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
from basic_code import load  # reuse the model-loading function used during training
from tqdm import tqdm
from natsort import natsorted  # to sort video directories

# Device setup
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
count = {"happy": 0,"angry": 0, "disgusted": 0, "fear": 0, "sad": 0, "contempt": 0, "surprised": 0}
correct_emo = {"happy": 0,"angry": 0, "disgusted": 0, "fear": 0, "sad": 0, "contempt": 0, "surprised": 0}
total_emo = {"happy": 0,"angry": 0, "disgusted": 0, "fear": 0, "sad": 0, "contempt": 0, "surprised": 0}

frame_correct = [0] * 16

# --- Label mapping (same order used during training) ---
label_to_emotion = {
    0: 'happy',
    1: 'angry',
    2: 'disgusted',
    3: 'fear',
    4: 'sad',
    5: 'contempt',
    6: 'surprised'
}

# --- Wrapper for handling video input (same as the code used during training) ---
class VideoClassifierWrapper(nn.Module):
    def __init__(self, base_model):
        super(VideoClassifierWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        # Process each frame as an individual image.
        x = x.view(B * T, C, H, W)
        logits = self.base_model(x)  # [B*T, num_classes]
        logits = logits.view(B, T, -1)  # [B, T, num_classes]
        probabilities = F.softmax(logits, dim=2)  # [B, T, num_classes]
        predicted_labels = probabilities.argmax(dim=2).cpu().tolist()[0]  # [T]
        # Video-level prediction: average the per-frame logits.
        logits = logits.mean(dim=1)

        return logits, predicted_labels

# --- Read the frames in a video folder ---
def read_video_frames(video_dir, num_frames, transform):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(valid_extensions)])
    if len(image_files) == 0:
        return None
    frames = []
    total = len(image_files)
    # Sample num_frames evenly (repeat the last frame if there aren't enough).
    if total >= num_frames:
        indices = [int(total / num_frames * i) for i in range(num_frames)]
    else:
        indices = list(range(total))
        while len(indices) < num_frames:
            indices.append(total - 1)
    for idx in indices:
        img_path = os.path.join(video_dir, image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if transform:
            image = transform(image)
        frames.append(image)
    video_tensor = torch.stack(frames)  # [num_frames, C, H, W]
    return video_tensor

# --- Predict the emotion for a single video ---
def predict_emotion_for_video(video_dir, model, transform, num_frames):
    video_tensor = read_video_frames(video_dir, num_frames, transform)
    if video_tensor is None:
        return None
    # Add a batch dimension for the model input: [1, num_frames, C, H, W]
    video_tensor = video_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, frame_labels = model(video_tensor)  # [1, num_classes]
        probabilities = F.softmax(logits, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
    return predicted_label, frame_labels

def main():
    parser = argparse.ArgumentParser(description="Predict video emotions and compute accuracy")
    parser.add_argument('--csv_file', type=str, required=True,
                        help="Path to the CSV file (columns: gt_emotion, ...)")
    parser.add_argument('--checkpoint', type=str, default="/path/to/SEVA/Emotion-FAN/checkpoints/checkpoint_epoch_12_baseline.pth",
                        help="Path to the checkpoint file (e.g. best_model.pth)")
    parser.add_argument('--num_frames', type=int, default=16,
                        help="Number of frames to sample from the video")
    args = parser.parse_args()

    # --- Preprocessing transform (same as during training) ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Build the model ---
    # Base architecture: ResNet18 (7 classes)
    _structure = models.resnet18(num_classes=7)
    _parameterDir = './Emotion-FAN/pretrain_model/Resnet18_FER+_pytorch.pth.tar'
    base_model = load.model_parameters(_structure, _parameterDir)
    # Wrap with the same wrapper used during training.
    model = VideoClassifierWrapper(base_model)
    model = model.to(DEVICE)
    cudnn.benchmark = True

    # --- Load the checkpoint (finetuned weights) ---
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # --- Load the CSV file (ground-truth emotion info) ---
    df = pd.read_csv(args.csv_file)
    # Add a predicted_emotion column if it doesn't exist yet (starts empty).
    if 'predicted_emotion' not in df.columns:
        df['predicted_emotion'] = ''

    # --- Sort the per-video subfolders under the frame parent directory ---
    csv_basename = os.path.basename(args.csv_file)  # filename only
    name_only = os.path.splitext(csv_basename)[0]
    frame_dir = os.path.join("./runs/mead_ours/faces", name_only)
    frame_dirs = natsorted([os.path.join(frame_dir, d) for d in os.listdir(frame_dir)
                             if os.path.isdir(os.path.join(frame_dir, d))])

    total_videos = min(len(df), len(frame_dirs))
    save_interval = 50
    processed_since_save = 0

    for idx in tqdm(range(total_videos), desc="Predicting emotions"):
        existing = df.at[idx, 'predicted_emotion']
        if pd.notna(existing) and str(existing).strip() != '':
            continue
        try:
            gt_emotion = df.iloc[idx]['gt_emotion'].strip().lower()
            video_dir = frame_dirs[idx]
            # Record 'neutral' in predicted_emotion for neutral rows too.
            if gt_emotion == 'neutral':
                df.at[idx, 'predicted_emotion'] = 'neutral'
                processed_since_save += 1
            else:
                predicted_label, frame_labels = predict_emotion_for_video(video_dir, model, transform, args.num_frames)
                if predicted_label is None:
                    tqdm.write(f"No valid frames found for video {video_dir}, skipping.")
                    df.at[idx, 'predicted_emotion'] = None
                    continue

                predicted_emotion = label_to_emotion[predicted_label]
                count[predicted_emotion] += 1
                df.at[idx, 'predicted_emotion'] = predicted_emotion

                if predicted_emotion == gt_emotion:
                    correct_emo[gt_emotion] += 1
                total_emo[gt_emotion] += 1
                for f_idx, f_label in enumerate(frame_labels):
                    if label_to_emotion[f_label] == gt_emotion:
                        frame_correct[f_idx] += 1

                tqdm.write(f"Video: {video_dir}\n  GT emotion: {gt_emotion}\n  Predicted emotion: {predicted_emotion}\n")
                processed_since_save += 1

            if processed_since_save >= save_interval:
                df.to_csv(args.csv_file, index=False)
                processed_since_save = 0
                tqdm.write(f"Checkpoint saved at row {idx}")
        except Exception as e:
            tqdm.write(f"{idx} {e}")
            continue

    # Final accuracy over the whole CSV, so a resumed run still reports
    # correctly (the per-emotion/per-frame stats below only cover rows
    # processed in this run, since frame-level labels aren't persisted).
    valid = df[df['predicted_emotion'].notna() & (df['predicted_emotion'].astype(str).str.strip() != '')]
    final_correct = (valid['gt_emotion'].str.strip().str.lower() == valid['predicted_emotion'].str.strip().str.lower()).sum()
    final_total = len(valid)
    accuracy = (final_correct / final_total) * 100 if final_total > 0 else 0
    print(f"\nTotal videos: {final_total}, Correct: {final_correct}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print("Prediction count per emotion (this run):", count)
    for emotion in total_emo:
        if total_emo[emotion] > 0:
            print(f"{emotion} (this run): {correct_emo[emotion]/total_emo[emotion]*100:.2f}% ({correct_emo[emotion]}/{total_emo[emotion]})")

    total_this_run = sum(total_emo.values())
    if total_this_run > 0:
        for f_idx, f_cor in enumerate(frame_correct):
            print(f'Frame {f_idx+1} accuracy (this run): {frame_correct[f_idx] / total_this_run * 100}%')

    # Update the CSV file (overwrite the original, or save as a new file).
    df.to_csv(args.csv_file, index=False)
    print(f"CSV file with prediction results saved: {args.csv_file}")

if __name__ == '__main__':
    main()
