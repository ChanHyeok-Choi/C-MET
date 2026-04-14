import argparse
import os, sys
sys.path.append('src')
sys.path.append('src/metavoice')
from huggingface_hub import hf_hub_download

from datetime import datetime
from pathlib import Path

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from src.EDTalk.networks.generator import Generator
from src.EDTalk.networks.utils import check_package_installed
from src.EDTalk.networks.audio_encoder import Audio2Lip
from src.connector import Connector_exp
from src.util import vid_preprocessing, save_video, load_model, img_preprocessing, get_mel, audio_preprocessing, conv_feat
from moviepy.editor import *
from src.dataset_emo12 import Dataset

import pickle
from funasr import AutoModel
import pandas as pd
import random

HF_REPO_ID = "coldhyuk/C-MET"
PRETRAINED_WEIGHT_FILES = ["Audio2Lip.pt", "EDTalk.pt", "EDTalk-V.pt"]


def ensure_pretrained_weights(pretrained_dir="./pretrained_weights"):
    """Download pretrained weights from HF Hub if not present locally."""
    os.makedirs(pretrained_dir, exist_ok=True)
    for filename in PRETRAINED_WEIGHT_FILES:
        local_path = os.path.join(pretrained_dir, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename} from Hugging Face...")
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"pretrained_weights/{filename}",
                local_dir=".",
            )


predefined_emo = ["angry", "contempt", "disgusted", "fear", "happy", "sad", "surprised"]
emo_map = {"charisma": "contempt", "empathy": "sad", "desire": "neutral", "envy": "neutral", "sarcasm": "neutral", "romance": "neutral"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sr", action="store_true")
    parser.add_argument("--save_np", action="store_true")

    parser.add_argument('--mode', type=str, default='mean', choices=['single', 'mean'], help='mode of the model to train')
    parser.add_argument('--direction', type=str, default='average', choices=['first', 'max', 'average', 'random', 'raw'], help='what value to use for the direction')
    parser.add_argument('--num_samples', type=int, default=10, help='number of samples to calculate a speaker direction')

    parser.add_argument('--connector_exp_path', type=str, default=None, help='path to the connector exp model')

    parser.add_argument('--audio_encoder', type=str, default='emotion2vec+large', help='emotion2vec or emotion2vec+large')

    parser.add_argument('--opt', type=str, default='A', choices=['A', 'V'], help='use A: audio or V: video as lip input')

    parser.add_argument("--source_path", type=str, default='asset/stephen_curry_crop.png')
    parser.add_argument("--audio_driving_path", type=str, default='test_data/stephen_curry.wav')
    parser.add_argument("--lip_driving_path", type=str, default='test_data/stephen_curry.mp4')
    parser.add_argument("--pose_driving_path", type=str, default='test_data/stephen_curry.mp4')
    parser.add_argument("--save_path", type=str, default='res/stephen_curry_ours.mp4')

    parser.add_argument("--neu_e2v_path", type=str, default=None)
    parser.add_argument("--emo_e2v_path", type=str, default=None)

    parser.add_argument("--emo", type=str, default='angry')
    parser.add_argument("--intensity", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    fix_seed(args.seed)
    ensure_pretrained_weights()

    config = OmegaConf.load(args.config)
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    device = args.device
    if device.__contains__("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")
    
    pretrained_EDTalk = OmegaConf.to_container(config.pretrained_EDTalk, resolve=True)
    projector_kwargs = OmegaConf.to_container(config.projector_kwargs, resolve=True)
    transformer_kwargs = OmegaConf.to_container(config.transformer_kwargs, resolve=True)
    T = transformer_kwargs['T']

    ############# model_init started #############

    # model_id = config["emotion2vec_model_id"]
    # model = AutoModel(
    #     model=model_id,
    #     hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
    # )

    audio2lip_model_path = config["audio2lip_model_path"]
    audio2lip = Audio2Lip().cuda()
    weight = torch.load(audio2lip_model_path, map_location=lambda storage, loc: storage)['audio2lip']
    audio2lip.load_state_dict(weight)
    audio2lip.eval()
    
    gen = Generator(pretrained_EDTalk['size'], 
                    style_dim=pretrained_EDTalk['latent_dim_style'], 
                    lip_dim=pretrained_EDTalk['latent_dim_lip'], 
                    pose_dim=pretrained_EDTalk['latent_dim_pose'], 
                    exp_dim=pretrained_EDTalk['latent_dim_exp'], 
                    channel_multiplier=pretrained_EDTalk['channel_multiplier']).to(device)
    weight = torch.load(pretrained_EDTalk['model_path'], map_location=lambda storage, loc: storage)['gen']
    gen.load_state_dict(weight)
    gen.eval()
    
    connector_exp = Connector_exp(projector_kwargs, transformer_kwargs, device).to(device)
    if args.connector_exp_path is None:
        connector_exp_path = config.connector_exp_path
    else:
        connector_exp_path = args.connector_exp_path
    weight_exp = torch.load(connector_exp_path, map_location=lambda storage, loc: storage)
    connector_exp.load_state_dict(weight_exp['state_dict'])
    connector_exp.eval()

    # val_dataset = Dataset('test',dataset_root='./dataset/MEAD/FPS25', 
    #                         T=T, mode=args.mode, direction=args.direction, 
    #                         num_samples=args.num_samples, audio_encoder=args.audio_encoder)
        
    ############# model_init finished #############    
    
    if args.neu_e2v_path is not None and args.emo_e2v_path is not None:
        neu_e2v = [os.path.join(args.neu_e2v_path, e2v) for e2v in os.listdir(args.neu_e2v_path)]
        emo_e2v = [os.path.join(args.emo_e2v_path, e2v) for e2v in os.listdir(args.emo_e2v_path)]

        random_neu = random.sample(neu_e2v, 10)
        random_emo = random.sample(emo_e2v, 10)
        neu = torch.stack([torch.from_numpy(np.load(i)).float() for i in random_neu])
        emo = torch.stack([torch.from_numpy(np.load(i)).float() for i in random_emo])

        neu = neu.mean(dim=0)
        emo = emo.mean(dim=0)
        e2v = emo - neu
    else:
        e2v, _, _ = val_dataset.get_raw_e2v(emotion1='neutral', emotion2=args.emo, intensity='level_'+str(int(args.intensity)))

    e2v = e2v.unsqueeze(0).unsqueeze(0).to(device)

    img_name = args.source_path
    img_source = img_preprocessing(img_name, args.size).cuda()

    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if args.opt == 'A':
        src_aud_path = args.audio_driving_path
        audio, audio_bs, audio_T = audio_preprocessing(src_aud_path)
        lip_vid_target = audio2lip(audio, audio_bs, audio_T)[0]
        lip_vid_target = conv_feat(lip_vid_target, k_size=3, sigma=1) # torch.Size([372, 500])
        lip_vid_target = lip_vid_target.to(device)
    elif args.opt == 'V':
        src_aud_path = args.audio_driving_path
        lip_vid_target, fps = vid_preprocessing(args.lip_driving_path)
        lip_vid_target = lip_vid_target.to(device)

    source_video_path = args.pose_driving_path
    pose_vid_target, fps = vid_preprocessing(source_video_path)
    pose_vid_target = pose_vid_target.to(device)

    len_pose = pose_vid_target.shape[1]
    if args.opt == 'A':
        lip_len = lip_vid_target.size(0)
    elif args.opt == 'V':
        lip_len = lip_vid_target.size(1)

    src_vid_target, fps = vid_preprocessing(source_video_path)
    src_vid_target = src_vid_target.to(device)
    src_vid_len = src_vid_target.shape[1]

    vid_len = src_vid_len - src_vid_len % T

    with torch.no_grad():
        batch_vid = src_vid_target.view(-1, src_vid_target.size(2), src_vid_target.size(3), src_vid_target.size(4))
        ED_neu, _, _ = gen.compute_alpha_D(batch_vid)
        ED_neu = ED_neu.unsqueeze(0).to(device)

    ED_ref_T = torch.zeros((1, T, ED_neu.size(2))).to(device)
    predicted_alpha_D_exp = []
    with torch.no_grad():
        for i in range(0, vid_len, T):
            ED_neu_T = ED_neu[:, i:i+T, :]
            pred_exp_dir, (_, _, _) = connector_exp(ED_ref_T, e2v, ED_neu_T)
            pred_exp = ED_neu_T.squeeze(0) + pred_exp_dir
            ED_ref_T = pred_exp_dir.unsqueeze(0)
            predicted_alpha_D_exp.append(pred_exp)
    predicted_alpha_D_exp = torch.cat(predicted_alpha_D_exp, dim=0)  # (vid_len, 10)
    predicted_alpha_D_exp = predicted_alpha_D_exp.unsqueeze(0)
    exp_vid_target = predicted_alpha_D_exp
    
    vid_target_recon = []

    exp_vid_target = exp_vid_target[:,:-20]
    while  exp_vid_target.shape[1]<lip_len:
        exp_vid_target = torch.cat([exp_vid_target, torch.flip(exp_vid_target, dims =[1])], dim=1)
    exp_vid_target = exp_vid_target[:lip_len]

    exp_len = exp_vid_target.shape[1]

    with torch.no_grad():
        for i in tqdm(range(lip_len)):
            if args.opt == 'A':
                img_target_lip = lip_vid_target[i:i+1]
            elif args.opt == 'V':
                img_target_lip = lip_vid_target[:, i, :, :, :]
            if i>=len_pose:
                img_target_pose = pose_vid_target[:, -1, :, :, :]
            else:
                img_target_pose = pose_vid_target[:, i, :, :, :]
            if i>=exp_len:
                alpha_D_exp = exp_vid_target[:, -1, :]
            else:
                alpha_D_exp = exp_vid_target[:, i, :]
            if args.opt == 'A':
                img_recon = gen.test_EDTalk_AV_use_exp_weight(img_source, img_target_lip, img_target_pose, alpha_D_exp, h_start=None)
            elif args.opt == 'V':
                img_recon = gen.test_EDTalk_V_use_exp_weight(img_source, img_target_lip, img_target_pose, alpha_D_exp, h_start=None)

            vid_target_recon.append(img_recon.unsqueeze(2))
        
    vid_target_recon = torch.cat(vid_target_recon, dim=2)
    
    temp_path = save_path.replace('.mp4','_temp.mp4')
    save_video(vid_target_recon, temp_path, fps)
    cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s" -y' % (temp_path, src_aud_path, save_path)
    os.system(cmd)
    os.remove(temp_path)
    
    if args.sr:
        enhance_256to512(save_path)

        

def enhance_256to512(save_path):
    if check_package_installed('gfpgan'):
        from src.EDTalk.face_sr.face_enhancer import enhancer_list
        import imageio
        
        temp_512_path = save_path.replace('.mp4','_512.mp4')

        # Super-resolution
        imageio.mimsave(temp_512_path + '.tmp.mp4', enhancer_list(save_path, method='gfpgan', bg_upsampler=None), fps=float(25), codec='libx264')
        
        # Merge audio and video
        video_clip = VideoFileClip(temp_512_path + '.tmp.mp4')
        audio_clip = AudioFileClip(save_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(temp_512_path, codec='libx264', audio_codec='aac')
        
        os.remove(temp_512_path + '.tmp.mp4')

    
if __name__ == "__main__":
    main()
