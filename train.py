import os
import argparse

from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from src.dataset_emo12 import Dataset
from src.connector import Connector_exp
from loss import *

from collections import defaultdict
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--mode', type=str, default='mean', choices=['single', 'mean'], help='mode of the model to train')
    parser.add_argument('--num_feats', type=int, default=10, help='number of expression features') 
    parser.add_argument('--direction', type=str, default='average', choices=['first', 'max', 'average', 'random', 'raw', 'all'], help='what value to use for the direction')
    parser.add_argument('--ID', type=str, default='same', choices=['same', 'diff'], help='ID type for training')
    parser.add_argument('--feature_type', type=str, default='ED', choices=['ED', 'PD_FGC'], help='feature type to use')

    parser.add_argument('--except_emotions', nargs='+', type=str, default=None, help='emotions to exclude from the training')

    parser.add_argument('--audio_encoder', type=str, default='emotion2vec+large', help='emotion2vec or emotion2vec+large')

    parser.add_argument('--train', type=str, default='bidir', help='unidir or bidir')

    parser.add_argument('--lambda_dir', type=float, default=0.05, help='weight for the direction loss')
    parser.add_argument('--lambda_cnt', type=float, default=0.1, help='weight for the contrastive loss')
    parser.add_argument('--scale_intensity', action='store_true', help='scale intensity in speech direction')
    parser.add_argument('--balance', type=str, default='focal_mse', choices=['mse', 'wmse', 'focal_mse', 'bmse'], help='balance method for the loss function')

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    return args


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, save_optimizer_state=True, prefix='', save_num=None):
    checkpoint_path = os.path.join(
        checkpoint_dir, "{}_epoch_{}_checkpoint_step{:09d}.pth".format(prefix, epoch, step))
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

    if save_num is not None:
        # remain only the last 20 checkpoints
        checkpoints = sorted(os.listdir(checkpoint_dir), key=lambda x: int(x.split('_')[-1].split('.')[0].replace('step', '')))
        if len(checkpoints) > save_num:
            for i in range(len(checkpoints) - save_num):
                os.remove(os.path.join(checkpoint_dir, checkpoints[i]))
                print("Removed checkpoint:", os.path.join(checkpoint_dir, checkpoints[i]))


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args):
    fix_seed(args.seed)
    config = OmegaConf.load(args.config)
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    device = args.device
    if device.__contains__("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")

    projector_kwargs = OmegaConf.to_container(config.projector_kwargs, resolve=True)
    transformer_kwargs = OmegaConf.to_container(config.transformer_kwargs, resolve=True)
    hidden_size = transformer_kwargs['hidden_size']
    T = transformer_kwargs['T']
    nhead = transformer_kwargs['nhead']
    nlayer = transformer_kwargs['nlayer']
    Project_name = config.Project_name + f'_MSE+{str(args.lambda_cnt)}Cnt+{str(args.lambda_dir)}Dir' + f'_{args.ID}ID' + f'_{args.train}'\
                        + f'_{nhead}H{nlayer}L_' + f'{hidden_size}' + '_' + args.mode\
                        + str(args.num_feats) + '_' + args.direction + f'_{args.except_emotions}_si{args.scale_intensity}_{args.balance}'
    
    checkpoint_interval = config.checkpoint_interval
    evaluate_interval = config.evaluate_interval
    checkpoint_dir = config.checkpoint_dir + '/' + Project_name
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    writer = SummaryWriter('tensorboard_runs/Project_{}'.format(Project_name))
    
    global_step, global_epoch = 0, 0

    # Load the dataset
    print("Loading the dataset...")
    batch_size = config.batch_size
    batch_size_val = config.batch_size_val
    num_workers = config.num_workers
    train_data_loader = torch.utils.data.DataLoader(
        Dataset('train', T=T, mode=args.mode, num_feats=args.num_feats,
         direction=args.direction, except_emotions=args.except_emotions, 
         audio_encoder=args.audio_encoder, scale_intensity=args.scale_intensity, ID=args.ID,
         feature_type=args.feature_type),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        Dataset('test', T=T, mode=args.mode, num_feats=args.num_feats,
        direction=args.direction, except_emotions=args.except_emotions, 
        audio_encoder=args.audio_encoder, scale_intensity=args.scale_intensity, ID=args.ID,
        feature_type=args.feature_type),
        batch_size=batch_size_val,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Define a connector model
    connector = Connector_exp(projector_kwargs, transformer_kwargs, device).to(device)
    
    # Define an optimizer
    # optimizer = torch.optim.AdamW(list(connector.parameters()) + list(criterion_BMC.parameters()), lr=config.lr, weight_decay=config.weight_decay)
    optimizer = torch.optim.AdamW(connector.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if args.balance == 'bmse':
        optimizer.add_param_group({'params': criterion_BMC.noise_sigma, 'lr': config.lr, 'name': 'noise_sigma'})
    
    # Train the model
    num_epochs = config.num_epochs
    for epoch in range(num_epochs):
        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        running_MSE_loss = 0.
        running_Cnt_loss = 0.
        running_Dir_loss = 0.
        for step, (e2v, ED_ref, ED_neu, ED_emo, emo_dir, emo_label, e2v_emo, e2v_neu) in pbar:
            B, T = emo_dir.size(0), emo_dir.size(1)
            e2v = e2v.cuda(non_blocking=True)  # (bs, 1, 768)
            ED_ref = ED_ref.cuda(non_blocking=True)  # (bs, T, 10)
            ED_neu = ED_neu.cuda(non_blocking=True)  # (bs, T, 10)
            ED_emo = ED_emo.cuda(non_blocking=True)  # (bs, T, 10)
            emo_dir = emo_dir.cuda(non_blocking=True)  # (bs, T, 10)
            emo_dir = emo_dir.view(-1, emo_dir.size(2))  # (bs*T, 10)
            emo_label = emo_label.cuda(non_blocking=True)  # (bs, T, 1)
            emo_label_cnt = emo_label[:, :1, :].view(-1)  # (bs)
            emo_label = emo_label.view(-1)  # (bs*T)
            e2v_emo = e2v_emo.cuda(non_blocking=True)  # (bs, 1, 768)
            e2v_neu = e2v_neu.cuda(non_blocking=True)  # (bs, 1, 768)
            
            if global_step % checkpoint_interval == 0:
                save_checkpoint(connector, optimizer, global_step, checkpoint_dir, global_epoch)
            if global_step % evaluate_interval == 0 or global_step == 100:
                with torch.no_grad():
                    evaluate(connector, val_data_loader, global_step, writer)
            
            connector.train()
            optimizer.zero_grad()
            
            # FORWARD PASS
            emo_hat, (ED_ref_hat, e2v_hat, ED_neu_hat) = connector(ED_ref, e2v, ED_neu)
            emo_hat_flipped, (ED_ref_hat_flipped, e2v_hat_flipped, ED_emo_hat) = connector(-ED_ref, -e2v, ED_emo)
            ED_dir_hat = ED_emo_hat - ED_neu_hat
            if args.train == 'bidir':
                if args.balance == 'focal_mse':
                    MSE_loss = weighted_focal_mse_loss(emo_hat, emo_dir, None, activate='sigmoid', beta=.2, gamma=1) + \
                                weighted_focal_mse_loss(emo_hat_flipped, -emo_dir, None, activate='sigmoid', beta=.2, gamma=1)  # Bi-dir            
                elif args.balance == 'bmse':
                    MSE_loss = criterion_BMC(emo_hat, emo_dir) + criterion_BMC(emo_hat_flipped, -emo_dir)  # Bi-dir
                elif args.balance == 'wmse':
                    MSE_loss = criterion_wMSE(emo_hat, emo_dir, emo_label) + criterion_wMSE(emo_hat_flipped, -emo_dir, emo_label)
                else:  # 'mse'
                    MSE_loss = criterion_wMSE(emo_hat, emo_dir) + criterion_wMSE(emo_hat_flipped, -emo_dir)
            elif args.train == 'unidir':
                if args.balance == 'focal_mse':
                    MSE_loss = weighted_focal_mse_loss(emo_hat, emo_dir, None, activate='sigmoid', beta=.2, gamma=1)  # Uni-dir
                elif args.balance == 'bmse':
                    MSE_loss = criterion_BMC(emo_hat, emo_dir)  # Uni-dir
                elif args.balance == 'wmse':
                    MSE_loss = criterion_wMSE(emo_hat, emo_dir, emo_label)
                else:
                    MSE_loss = criterion_wMSE(emo_hat, emo_dir)  # Uni-dir
            else:
                raise ValueError("Invalid training mode. Choose 'unidir' or 'bidir'.")
            Cnt_v2a_loss = criterion_Cnt(torch.mean(-ED_dir_hat, dim=1), e2v_hat.squeeze(1), emo_label_cnt)
            Cnt_a2v_loss = criterion_Cnt(torch.mean(ED_dir_hat, dim=1), e2v_hat_flipped.squeeze(1), emo_label_cnt)
            # Cnt_v2a_loss = criterion_Cnt(torch.mean(ED_neu_hat, dim=1), e2v_emo.squeeze(1), emo_label_cnt)
            # Cnt_a2v_loss = criterion_Cnt(torch.mean(ED_emo_hat, dim=1), e2v_neu.squeeze(1), emo_label_cnt)
            Cnt_loss = (Cnt_v2a_loss + Cnt_a2v_loss) / 2
            Dir_loss = criterion_Dir(emo_hat, emo_hat_flipped)
            loss = MSE_loss + args.lambda_cnt * (Cnt_loss) + args.lambda_dir * (Dir_loss)

            # BACKWARD PASS
            loss.backward()
            optimizer.step()
            
            # Update the running loss
            running_MSE_loss += MSE_loss.item()
            running_Cnt_loss += Cnt_loss.item()
            running_Dir_loss += Dir_loss.item()

            pbar.set_description('epoch: %d step: %d running_MSE_loss: %.4f running_Cnt_loss: %.4f running_Dir_loss: %.4f'
                                 % (global_epoch, global_step, running_MSE_loss / (step + 1), running_Cnt_loss / (step + 1), running_Dir_loss / (step + 1)))
            
            writer.add_scalar('running_MSE_loss', running_MSE_loss / (step + 1), global_step)
            writer.add_scalar('running_Cnt_loss', running_Cnt_loss / (step + 1), global_step)
            writer.add_scalar('running_Dir_loss', running_Dir_loss / (step + 1), global_step)

            global_step += 1
        global_epoch += 1
        
        
def evaluate(connector, val_data_loader, global_step, writer):
    connector.eval()
    eval_epochs = 25
    print('Evaluating model for {} epochs'.format(eval_epochs))
    
    eval_MSE_loss = 0.
    eval_EC_loss = 0.
    eval_Vel_loss = 0.
    eval_Dir_loss = 0.
    eval_Cnt_loss = 0.
    count = 0

    for epoch in tqdm(range(eval_epochs), total=eval_epochs):
        prog_bar = enumerate(val_data_loader)
        for step, (e2v, ED_ref, ED_neu, ED_emo, emo_dir, emo_label, e2v_emo, e2v_neu) in prog_bar:
            B, T = emo_dir.size(0), emo_dir.size(1)
            e2v = e2v.cuda(non_blocking=True)
            ED_ref = ED_ref.cuda(non_blocking=True)
            ED_neu = ED_neu.cuda(non_blocking=True)
            ED_emo = ED_emo.cuda(non_blocking=True)
            emo_dir = emo_dir.cuda(non_blocking=True).view(-1, emo_dir.size(2))  # (B*T, D)
            emo_label = emo_label.cuda(non_blocking=True)
            emo_label_cnt = emo_label[:, :1, :].view(-1)  # (B)
            emo_label = emo_label.view(-1)  # (B*T)
            e2v_emo = e2v_emo.cuda(non_blocking=True)  # (bs, 1, 768)
            e2v_neu = e2v_neu.cuda(non_blocking=True)  # (bs, 1, 768)

            # forward pass
            emo_hat, (ED_ref_hat, e2v_hat, ED_neu_hat) = connector(ED_ref, e2v, ED_neu)
            emo_hat_flipped, (ED_ref_hat_flipped, e2v_hat_flipped, ED_emo_hat) = connector(-ED_ref, -e2v, ED_emo)
            ED_dir_hat = ED_emo_hat - ED_neu_hat

            # 전체 loss
            if args.train == 'bidir':
                if args.balance == 'focal_mse':
                    eval_MSE_loss += weighted_focal_mse_loss(emo_hat, emo_dir, None, activate='sigmoid', beta=.2, gamma=1).item() + \
                                    weighted_focal_mse_loss(emo_hat_flipped, -emo_dir, None, activate='sigmoid', beta=.2, gamma=1).item()  # Bi-dir 
                elif args.balance == 'bmse':
                    eval_MSE_loss += criterion_BMC(emo_hat, emo_dir).item() + criterion_BMC(emo_hat_flipped, -emo_dir).item()  # Bi-dir
                elif args.balance == 'wmse':
                    eval_MSE_loss += criterion_wMSE(emo_hat, emo_dir, emo_label).item() + criterion_wMSE(emo_hat_flipped, -emo_dir, emo_label).item()
                else:
                    eval_MSE_loss += criterion_wMSE(emo_hat, emo_dir).item() + criterion_wMSE(emo_hat_flipped, -emo_dir).item()  # Bi-dir
            elif args.train == 'unidir':
                if args.balance == 'focal_mse':
                    eval_MSE_loss += weighted_focal_mse_loss(emo_hat, emo_dir, None, activate='sigmoid', beta=.2, gamma=1).item()  # Uni-dir
                elif args.balance == 'bmse':
                    eval_MSE_loss += criterion_BMC(emo_hat, emo_dir).item()  # Uni-dir
                elif args.balance == 'wmse':
                    eval_MSE_loss += criterion_wMSE(emo_hat, emo_dir, emo_label).item()
                else:
                    eval_MSE_loss += criterion_wMSE(emo_hat, emo_dir).item()  # Uni-dir
            else:
                raise ValueError("Invalid training mode. Choose 'unidir' or 'bidir'.")
            eval_EC_loss += criterion_wEC(emo_hat, emo_dir, emo_label).item() + criterion_wEC(emo_hat_flipped, -emo_dir, emo_label).item()
            eval_Vel_loss += criterion_Vel(
                emo_hat.view(B, T, emo_hat.size(1)),
                emo_dir.view(B, T, emo_dir.size(1))
            ).item()
            eval_Dir_loss += criterion_Dir(emo_hat, emo_hat_flipped).item()
            Cnt_v2a_loss = criterion_Cnt(torch.mean(-ED_dir_hat, dim=1), e2v_hat.squeeze(1), emo_label_cnt)
            Cnt_a2v_loss = criterion_Cnt(torch.mean(ED_dir_hat, dim=1), e2v_hat_flipped.squeeze(1), emo_label_cnt)
            eval_Cnt_loss += (Cnt_v2a_loss.item() + Cnt_a2v_loss.item()) / 2
            count += 1

    # 기록
    writer.add_scalar('eval_MSE_loss', eval_MSE_loss / count, global_step)
    writer.add_scalar('eval_EC_loss', eval_EC_loss / count, global_step)
    writer.add_scalar('eval_Vel_loss', eval_Vel_loss / count, global_step)
    writer.add_scalar('eval_Dir_loss', eval_Dir_loss / count, global_step)
    writer.add_scalar('eval_Cnt_loss', eval_Cnt_loss / count, global_step)
    print('eval_MSE_loss', eval_MSE_loss / count, 'global_step:', global_step)
    print('eval_EC_loss', eval_EC_loss / count, 'global_step:', global_step)
    print('eval_Vel_loss', eval_Vel_loss / count, 'global_step:', global_step)
    print('eval_Dir_loss', eval_Dir_loss / count, 'global_step:', global_step)
    print('eval_Cnt_loss', eval_Cnt_loss / count, 'global_step:', global_step)


if __name__ == '__main__':
    args = parse_args()
    train(args)