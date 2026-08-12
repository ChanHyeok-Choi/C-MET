import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from basic_code import load  # model-loading function provided by the existing code
import torch.backends.cudnn as cudnn
import logging

# --- Logger setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

# --- util module (checkpoint saving, accuracy calculation) ---
class util:
    @staticmethod
    def save_checkpoint(state, checkpoint_dir, at_type='baseline'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{state["epoch"]}_{at_type}.pth')
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

# --- Device ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Emotion label mapping (FER+ convention, 7 classes) ---
# emotion_to_label = {
#     'happy': 0,
#     'angry': 1,
#     'disgusted': 2,
#     'fear': 3,
#     'sad': 4,
#     'contempt': 5,
#     'surprised': 6
# }
# emotion_to_label = {
#     'happy': 0,
#     'angry': 1,
#     'disgusted': 2,
#     'fear': 3,
#     'sad': 4,
#     'calm': 5,
#     'surprised': 6,
# }
emotion_to_label = {
    'hap': 0,
    'ang': 1,
    'dis': 2,
    'fea': 3,
    'sad': 4
}
label_to_emotion = {v: k for k, v in emotion_to_label.items()}

# --- Dataset class ---
class FaceDirDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, num_frames=16, transform=None):
        self.data = pd.read_csv(csv_file)
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def read_face_dir(self, face_dir):
        valid_extensions = ('.jpg', '.jpeg', '.png')
        image_files = sorted([f for f in os.listdir(face_dir) if f.lower().endswith(valid_extensions)])
        if len(image_files) == 0:
            return None
        frames = []
        total = len(image_files)
        if total >= self.num_frames:
            indices = [int(total / self.num_frames * i) for i in range(self.num_frames)]
        else:
            indices = list(range(total))
            while len(indices) < self.num_frames:
                indices.append(total - 1)
        for idx in indices:
            img_path = os.path.join(face_dir, image_files[idx])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        video_tensor = torch.stack(frames)  # [num_frames, C, H, W]
        return video_tensor

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emotion = row['gt_emotion'].strip().lower()
        face_dir = row['face_dir'].strip()
        label = emotion_to_label.get(emotion, -1)
        if label == -1:
            raise ValueError(f"Invalid emotion label found: '{emotion}' in row {idx}")
        video_tensor = self.read_face_dir(face_dir)
        if video_tensor is None:
            video_tensor = torch.zeros(self.num_frames, 3, 224, 224)
        return video_tensor, label

def split_csv(csv_file, train_csv='train.csv', test_csv='test.csv', test_size=0.2, random_state=42):
    df = pd.read_csv(csv_file)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    return train_csv, test_csv

# --- Wrapper for handling video input ---
class VideoClassifierWrapper(nn.Module):
    def __init__(self, base_model):
        super(VideoClassifierWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        # Reshape to process each frame as an individual image.
        x = x.view(B * T, C, H, W)
        logits = self.base_model(x)  # [B*T, num_classes]
        logits = logits.view(B, T, -1)
        # Video-level prediction: average the per-frame logits.
        logits = logits.mean(dim=1)
        return logits

# --- Train function ---
def train(train_loader, model, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (videos, labels) in enumerate(train_loader):
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * videos.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 200 == 0:
            logger.info(f"Epoch [{epoch}][{i}/{len(train_loader)}]: Loss: {loss.item():.4f}")
    avg_loss = running_loss / total
    acc = correct / total * 100
    logger.info(f"Train Epoch {epoch}: Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

# --- Validation function ---
def val(val_loader, model):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(videos)
            loss = F.cross_entropy(outputs, labels)
            running_loss += loss.item() * videos.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    avg_loss = running_loss / total
    acc = correct / total * 100
    logger.info(f"Validation: Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
    return acc

# --- Main training function ---
def main_training(train_csv, test_csv, num_epochs=60, batch_size=8, learning_rate=1e-1, num_frames=16,
                  checkpoint_dir='./checkpoints2', save_interval=1, evaluate=False):
    os.makedirs(checkpoint_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FaceDirDataset(train_csv, num_frames=num_frames, transform=transform)
    test_dataset = FaceDirDataset(test_csv, num_frames=num_frames, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    ''' Load model '''
    _structure = models.resnet18(num_classes=5)
    _parameterDir = './Emotion-FAN/pretrain_model/Resnet18_FER+_pytorch.pth.tar'
    base_model = load.model_parameters(_structure, _parameterDir)
    # Wrap it so it can handle video input ([B, T, C, H, W]).
    model = VideoClassifierWrapper(base_model)
    model = model.to(DEVICE)

    ''' Loss & Optimizer '''
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                learning_rate, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    cudnn.benchmark = True

    best_acc = 0.0
    if evaluate:
        logger.info(f'args.evaluate: {evaluate}')
        acc = val(val_loader, model)
        return

    logger.info(f'Baseline dataset, learning rate: {learning_rate}')
    for epoch in range(num_epochs):
        train(train_loader, model, optimizer, epoch)
        acc_epoch = val(val_loader, model)
        if acc_epoch > best_acc:
            logger.info('Better model!')
            best_acc = acc_epoch
            util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'accuracy': acc_epoch,
            }, checkpoint_dir, at_type='baseline')
        lr_scheduler.step()
        logger.info(f"Epoch: {epoch+1} Learning rate: {optimizer.param_groups[0]['lr']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video-level emotion classifier using CSV file with face_dir')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the training CSV (face_dir + gt_emotion columns)')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the validation/test CSV (face_dir + gt_emotion columns)')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to sample per video')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints2', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=1, help='Interval (in epochs) for saving model checkpoints')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model only')
    args = parser.parse_args()

    main_training(train_csv=args.train_csv, test_csv=args.test_csv, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                  num_frames=args.num_frames, checkpoint_dir=args.checkpoint_dir, save_interval=args.save_interval, evaluate=args.evaluate)
