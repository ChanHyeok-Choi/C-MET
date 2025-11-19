import os
import sys
sys.path.append("src")
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import argparse
from tqdm import tqdm
from glob import glob

import torch

from src.EDTalk.networks.generator import Generator as EDTalk_Generator
from src.DPE.networks.generator import Generator as DPE_Generator
from src.util import vid_preprocessing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--process_num", type=int, default=4)
    parser.add_argument("--angle", type=str, default='front')
    parser.add_argument("--mode", type=str, default='full', choices=['full', 'batch'])
    parser.add_argument("--type", type=str, default='EDTalk', choices=['EDTalk', 'DPE', 'PD-FGC'])
    
    args = parser.parse_args()
    return args


def main(args):
    import os
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("looking up paths.... from", args.data_root) 

    ## Second, convert npy to EDTalk expression embedding

    # MEAD
    # vidlist = glob(path.join(args.data_root, f'*/{args.angle}/*/*/*.mp4'))
    # vidlist.sort()
    # print("total vid files:", len(vidlist))

    # CREMA-D
    vidlist = glob(os.path.join(args.data_root, '*.mp4'))
    vidlist.sort()
    print("total vid files:", len(vidlist))
    
    if args.type == 'EDTalk':
        pretrained_EDTalk = 'pretrained_weights/EDTalk.pt'
        gen = EDTalk_Generator(size=256, style_dim=512, lip_dim=20, pose_dim=6, exp_dim=10, channel_multiplier=1).cuda()
        weight = torch.load(pretrained_EDTalk, map_location=lambda storage, loc: storage)['gen']
        gen.load_state_dict(weight)
        gen.eval()

        for vid_path in tqdm(vidlist, total=len(vidlist)):
            _, dataset, CREMA_D, FPS25, video_name = vid_path.split('/')
            num, ID, emotion, intensity = video_name.replace('.mp4', '').split('_')
            num = num + '.mp4'
            fulldir = os.path.join(args.data_root, ID, emotion, intensity)  # /data1/MEAD/VIDEO
            os.makedirs(fulldir, exist_ok=True)

            ED_exp_path = os.path.join(fulldir, num.replace('.mp4', '_ED_exp.npy'))
            ED_pose_path = os.path.join(fulldir, num.replace('.mp4', '_ED_pose.npy'))
            ED_lip_path = os.path.join(fulldir, num.replace('.mp4', '_ED_lip.npy'))
            if os.path.exists(ED_exp_path) and os.path.exists(ED_pose_path) and os.path.exists(ED_lip_path):
                continue
            try:
                vid, fps = vid_preprocessing(vid_path)
                vid = vid.to(device)
                T = vid.shape[1]
                with torch.no_grad():
                    # Cuda out of memory error occurs, so manually process the video in batch.
                    if args.mode == 'batch':
                        batch_size = 100
                        ED_exp_list, ED_pose_list, ED_lip_list = [], [], []
                        for i in range(0, T, batch_size):
                            vid_batch = vid[:, i:i+batch_size, :, :, :].view(-1, vid.size(2), vid.size(3), vid.size(4))
                            ED_exp, ED_pose, ED_lip = gen.compute_alpha_D(vid_batch)
                            ED_exp = ED_exp.cpu().numpy()
                            ED_pose = ED_pose.cpu().numpy()
                            ED_lip = ED_lip.cpu().numpy()
                            ED_exp_list.append(ED_exp)
                            ED_pose_list.append(ED_pose)
                            ED_lip_list.append(ED_lip)
                        ED_exp = np.concatenate(ED_exp_list, axis=0)
                        ED_pose = np.concatenate(ED_pose_list, axis=0)
                        ED_lip = np.concatenate(ED_lip_list, axis=0)
                    elif args.mode == 'full':
                        vid = vid.view(-1, vid.size(2), vid.size(3), vid.size(4))  # (bs*T, C, H, W)
                        ED_exp, ED_pose, ED_lip = gen.compute_alpha_D(vid)
                        ED_exp = ED_exp.cpu().numpy()
                        ED_pose = ED_pose.cpu().numpy()
                        ED_lip = ED_lip.cpu().numpy()
                    np.save(ED_exp_path, ED_exp) if not os.path.exists(ED_exp_path) else None
                    np.save(ED_pose_path, ED_pose) if not os.path.exists(ED_pose_path) else None
                    np.save(ED_lip_path, ED_lip) if not os.path.exists(ED_lip_path) else None
                del vid
                torch.cuda.empty_cache()
            except Exception as e:
                print("Error in", vid_path, ":", e)
                continue

    elif args.type == 'PD-FGC':
        import os, sys
        sys.path.append('src')
        sys.path.append('src/PD_FGC')
        from src.PD_FGC.lib.config.config import cfg
        from src.PD_FGC.lib.inferencer import Tester
        from src.util import conv_feat

        tester = Tester(cfg)
        tester.reset_cfg(cfg)

        print("model loaded!")

        for vid_path in tqdm(vidlist, total=len(vidlist)):
            _, MEAD, FPS25, ID, angle, emotion, intensity, num = vid_path.split('/')
            fulldir = path.join(args.data_root, ID, angle, emotion, intensity)
            os.makedirs(fulldir, exist_ok=True)

            pose_path = os.path.join(fulldir, num.replace('.mp4', '_PD_FGC_pose.npy'))
            eye_path = os.path.join(fulldir, num.replace('.mp4', '_PD_FGC_eye.npy'))
            emo_path = os.path.join(fulldir, num.replace('.mp4', '_PD_FGC_emo.npy'))
            mouth_path = os.path.join(fulldir, num.replace('.mp4', '_PD_FGC_mouth.npy'))
            if os.path.exists(pose_path) and os.path.exists(eye_path) and os.path.exists(emo_path) and os.path.exists(mouth_path):
                continue
            try:
                wav = tester.dataset.audio.read_audio(vid_path.replace('.mp4', '.wav'))
                tester.dataset.spectrogram = tester.dataset.audio.audio_to_spectrogram(wav)

                tester.dataset.target_frame_inds = np.arange(2, len(tester.dataset.spectrogram) // tester.dataset.audio.num_bins_per_frame - 2)
                tester.dataset.audio_inds = tester.dataset.frame2audio_indexs(tester.dataset.target_frame_inds)
                spectrograms = []
                for index in range(tester.dataset.audio_inds.shape[0]):
                    mel_index = tester.dataset.audio_inds[index]
                    spectrogram = tester.dataset.load_spectrogram(mel_index)
                    spectrograms.append(spectrogram)
                spectrograms = torch.stack(spectrograms, 0)

                vid = tester.dataset.load_vid(vid_path)
                vid = [tester.dataset.to_Tensor(vid[i]) for i in range(len(vid))]
                vid = torch.stack(vid, 0)
                vid = vid.to(device)  # (T, C, H, W)

                # id_img = id_img.cuda()
                spectrograms = spectrograms.cuda()
                driving_imgs = [vid, vid, vid]

                emo_mem = None

                total_size = spectrograms.shape[0]
                dri_len0 = driving_imgs[0].shape[0]
                dri_len1 = driving_imgs[1].shape[0]
                dri_len2 = driving_imgs[2].shape[0]
                with torch.no_grad():
                    ### net_appearance forward
                    
                    # id_feature, id_scores = tester.net_appearance(id_img)

                    mouth_feat_list = []
                    pose_feat_list = []
                    eye_feat_list = []
                    emo_feat_list = []
                    # print(total_size)
                    for iters in range(int(round(total_size / tester.batch_size + 0.4999999))):
                        spectrogram = spectrograms[iters * tester.batch_size:min((iters + 1) * tester.batch_size, total_size)]
                        # print(spectrogram.shape)
                        A_mouth_feature, A_mouth_embed = tester.net_audio.forward(spectrogram)
                        A_mouth_embed = A_mouth_embed * 1.5
                        mouth_feat_list.append(A_mouth_embed)

                        dri_start0 = (iters * tester.batch_size) % dri_len0
                        dri_end0 = ((iters + 1) * tester.batch_size) % dri_len0
                        if dri_start0 < dri_end0:
                            driving_img0 = driving_imgs[0][dri_start0:dri_end0]
                        else:
                            driving_img0 = torch.cat([driving_imgs[0][dri_start0:], driving_imgs[0][:dri_end0]], 0)
                        driving_img0 = driving_img0[:min(tester.batch_size, abs(total_size - iters * tester.batch_size))]
                        
                        dri_start1 = (iters * tester.batch_size) % dri_len1
                        dri_end1 = ((iters + 1) * tester.batch_size) % dri_len1
                        if dri_start1 < dri_end1:
                            driving_img1 = driving_imgs[1][dri_start1:dri_end1]
                        else:
                            driving_img1 = torch.cat([driving_imgs[1][dri_start1:], driving_imgs[1][:dri_end1]], 0)
                        driving_img1 = driving_img1[:min(tester.batch_size, abs(total_size - iters * tester.batch_size))]

                        dri_start2 = (iters * tester.batch_size) % dri_len2
                        dri_end2 = ((iters + 1) * tester.batch_size) % dri_len2
                        if dri_start2 < dri_end2:
                            driving_img2 = driving_imgs[2][dri_start2:dri_end2]
                        else:
                            driving_img2 = torch.cat([driving_imgs[2][dri_start2:], driving_imgs[2][:dri_end2]], 0)
                        driving_img2 = driving_img2[:min(tester.batch_size, abs(total_size - iters * tester.batch_size))]

                        V_headpose_embed, V_eye_embed, emo_feat, mouth_feat = tester.net_motion(torch.cat([driving_img0, driving_img1, driving_img2], 0))
                        
                        pose_feat_list.append(V_headpose_embed[:A_mouth_embed.shape[0]])
                        eye_feat_list.append(V_eye_embed[A_mouth_embed.shape[0]*2:])
                        emo_feat_list.append(emo_feat[A_mouth_embed.shape[0]:A_mouth_embed.shape[0]*2])

                    mouth_feat_list = torch.cat(mouth_feat_list, 0)
                    pose_feat_list = torch.cat(pose_feat_list, 0)
                    eye_feat_list = torch.cat(eye_feat_list, 0)
                    emo_feat_list = torch.cat(emo_feat_list, 0)

                    mouth_feat = mouth_feat_list.cpu().numpy()
                    pose_feat = pose_feat_list.cpu().numpy()
                    eye_feat = eye_feat_list.cpu().numpy()
                    emo_feat = emo_feat_list.cpu().numpy()
                    
                    np.save(pose_path, pose_feat) if not os.path.exists(pose_path) else None
                    np.save(eye_path, eye_feat) if not os.path.exists(eye_path) else None
                    np.save(emo_path, emo_feat) if not os.path.exists(emo_path) else None
                    np.save(mouth_path, mouth_feat) if not os.path.exists(mouth_path) else None

                # T = vid.shape[0]
                # with torch.no_grad():
                #     # Cuda out of memory error occurs, so manually process the video in batch.
                #     if args.mode == 'batch':
                #         batch_size = 100
                #         pose_list, eye_list, emo_list, mouth_list = [], [], [], []
                #         for i in range(0, T, batch_size):
                #             vid_batch = vid[i:i+batch_size, :, :, :]
                #             pose, eye, emo, mouth = net_motion(torch.cat([vid_batch, vid_batch, vid_batch], 0))
                #             pose, eye, emo, mouth = pose.cpu().numpy(), eye.cpu().numpy(), emo.cpu().numpy(), mouth.cpu().numpy()
                #             pose_list.append(pose)
                #             eye_list.append(eye)
                #             emo_list.append(emo)
                #             mouth_list.append(mouth)
                #         pose = np.concatenate(pose_list, axis=0)
                #         eye = np.concatenate(eye_list, axis=0)
                #         emo = np.concatenate(emo_list, axis=0)
                #         mouth = np.concatenate(mouth_list, axis=0)
                #     np.save(pose_path, pose) if not os.path.exists(pose_path) else None
                #     np.save(eye_path, eye) if not os.path.exists(eye_path) else None
                #     np.save(emo_path, emo) if not os.path.exists(emo_path) else None
                #     np.save(mouth_path, mouth) if not os.path.exists(mouth_path) else None
                # del vid
                # torch.cuda.empty_cache()
            except Exception as e:
                print("Error in", vid_path, ":", e)
                continue
    else:
        raise NotImplementError


if __name__ == '__main__':
    args = parse_args()
    main(args)