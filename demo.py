import os
import sys
import random
import shutil

# MPS 미지원 연산(torch.qr 등)을 CPU로 자동 폴백 — torch import 전에 설정해야 적용됨
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.append('src')

import cv2
import imageio
import numpy as np
import torch
import face_alignment
import gradio as gr
from omegaconf import OmegaConf
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from moviepy.editor import AudioFileClip, VideoFileClip

from src.EDTalk.networks.generator import Generator
from src.EDTalk.networks.audio_encoder import Audio2Lip
from src.connector import Connector_exp
from src.util import (
    vid_preprocessing,
    save_video,
    img_preprocessing,
    audio_preprocessing,
    conv_feat,
)
from src.EDTalk.networks.utils import check_package_installed
from src.voice_clone import (
    is_vc_available,
    load_voice_cloning_models,
    transcribe_audio,
    clone_voice_with_emotion,
)

# ---------------------------------------------------------------------------
# Emotion catalogue
# ---------------------------------------------------------------------------

EMOTIONS = [
    ("angry",       "MEAD",   "audios/MEAD/angry/emotion2vec+large_features"),
    ("contempt",    "MEAD",   "audios/MEAD/contempt/emotion2vec+large_features"),
    ("disgusted",   "MEAD",   "audios/MEAD/disgusted/emotion2vec+large_features"),
    ("fear",        "MEAD",   "audios/MEAD/fear/emotion2vec+large_features"),
    ("happy",       "MEAD",   "audios/MEAD/happy/emotion2vec+large_features"),
    ("sad",         "MEAD",   "audios/MEAD/sad/emotion2vec+large_features"),
    ("surprised",   "MEAD",   "audios/MEAD/surprised/emotion2vec+large_features"),
    ("charismatic", "Gemini", "audios/gemini/charismatic/emotion2vec+large_features"),
    ("desirous",    "Gemini", "audios/gemini/desirous/emotion2vec+large_features"),
    ("empathetic",  "Gemini", "audios/gemini/empathetic/emotion2vec+large_features"),
    ("envious",     "Gemini", "audios/gemini/envious/emotion2vec+large_features"),
    ("romantic",    "Gemini", "audios/gemini/romantic/emotion2vec+large_features"),
    ("sarcastic",   "Gemini", "audios/gemini/sarcastic/emotion2vec+large_features"),
    ("sarcastic",   "MELD",   "audios/MELD/sarcastic/emotion2vec+large_features"),
]

NEU_E2V_PATH = "audios/MEAD/neutral/emotion2vec+large_features"

HF_REPO_ID = "coldhyuk/C-MET"
PRETRAINED_WEIGHT_FILES = ["Audio2Lip.pt", "EDTalk.pt", "EDTalk-V.pt"]
HF_CHECKPOINT_FILENAME = "checkpoints/_epoch_2105_checkpoint_step000200000.pth"

TMP_DIR    = "tmp_demo"
TMP_VC_DIR = "tmp_vc"

SAMPLE_VIDEO_DIR = "asset/video"
SAMPLE_VIDEOS = {
    name: os.path.join(SAMPLE_VIDEO_DIR, name)
    for name in sorted(os.listdir(SAMPLE_VIDEO_DIR))
    if name.endswith(".mp4")
} if os.path.isdir(SAMPLE_VIDEO_DIR) else {}

SAMPLE_IDENTITY_DIR = "asset/identity"
SAMPLE_IDENTITY = {
    name: os.path.join(SAMPLE_IDENTITY_DIR, name)
    for name in sorted(os.listdir(SAMPLE_IDENTITY_DIR))
    if name.lower().endswith((".png", ".jpg", ".jpeg"))
} if os.path.isdir(SAMPLE_IDENTITY_DIR) else {}

SAMPLE_AUDIO_DIR = "asset/audio"
SAMPLE_AUDIO = {
    name: os.path.join(SAMPLE_AUDIO_DIR, name)
    for name in sorted(os.listdir(SAMPLE_AUDIO_DIR))
    if name.lower().endswith(".wav")
} if os.path.isdir(SAMPLE_AUDIO_DIR) else {}

_SRC_DEFAULT = "기본값 (입력 비디오에서)"

# Build unique display names and a lookup map
EMOTION_DISPLAY_NAMES = []
EMOTION_DISPLAY_MAP = {}  # display_name -> (name, group, e2v_path)

for _name, _group, _path in EMOTIONS:
    _disp = _name if _group == "MEAD" else f"{_name} [{_group}]"
    EMOTION_DISPLAY_NAMES.append(_disp)
    EMOTION_DISPLAY_MAP[_disp] = (_name, _group, _path)

# ---------------------------------------------------------------------------
# Global model state — populated once by load_models() at startup
# ---------------------------------------------------------------------------

_MODELS: dict = {}


def get_device() -> str:
    """Return the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_fa_device() -> str:
    """Device for face_alignment (MPS not reliably supported)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Weight downloads
# ---------------------------------------------------------------------------

def ensure_pretrained_weights(pretrained_dir: str = "./pretrained_weights"):
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


def ensure_checkpoint(connector_exp_path: str):
    if not os.path.exists(connector_exp_path):
        print("Downloading connector checkpoint from Hugging Face...")
        os.makedirs(os.path.dirname(connector_exp_path), exist_ok=True)
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_CHECKPOINT_FILENAME,
            local_dir=".",
        )


# ---------------------------------------------------------------------------
# Model loading — called once at startup, result stored in _MODELS
# ---------------------------------------------------------------------------

def load_models(config_path: str = "./configs/inference.yaml"):
    global _MODELS
    device = get_device()
    print(f"Loading models on {device}...")

    config = OmegaConf.load(config_path)
    ensure_pretrained_weights()
    ensure_checkpoint(config.connector_exp_path)

    pretrained_EDTalk  = OmegaConf.to_container(config.pretrained_EDTalk,  resolve=True)
    projector_kwargs   = OmegaConf.to_container(config.projector_kwargs,   resolve=True)
    transformer_kwargs = OmegaConf.to_container(config.transformer_kwargs, resolve=True)

    audio2lip = Audio2Lip().to(device)
    w = torch.load(config.audio2lip_model_path, map_location=lambda s, l: s)["audio2lip"]
    audio2lip.load_state_dict(w)
    audio2lip.eval()

    gen = Generator(
        pretrained_EDTalk["size"],
        style_dim=pretrained_EDTalk["latent_dim_style"],
        lip_dim=pretrained_EDTalk["latent_dim_lip"],
        pose_dim=pretrained_EDTalk["latent_dim_pose"],
        exp_dim=pretrained_EDTalk["latent_dim_exp"],
        channel_multiplier=pretrained_EDTalk["channel_multiplier"],
    ).to(device)
    w = torch.load(pretrained_EDTalk["model_path"], map_location=lambda s, l: s)["gen"]
    gen.load_state_dict(w)
    gen.eval()

    connector_exp = Connector_exp(projector_kwargs, transformer_kwargs, device).to(device)
    w = torch.load(config.connector_exp_path, map_location=lambda s, l: s)
    connector_exp.load_state_dict(w["state_dict"])
    connector_exp.eval()

    _MODELS = {
        "audio2lip":     audio2lip,
        "gen":           gen,
        "connector_exp": connector_exp,
        "config":        config,
        "device":        device,
        "T":             transformer_kwargs["T"],
    }
    print("Models ready.")


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _detect_bboxes(frame_rgb, fa, max_size: int = 640):
    h, w = frame_rgb.shape[:2]
    if max(h, w) > max_size:
        scale = max(h, w) / max_size
        small = cv2.resize(frame_rgb, (int(w / scale), int(h / scale)))
    else:
        scale = 1.0
        small = frame_rgb
    raw = fa.face_detector.detect_from_image(small[..., ::-1])
    if len(raw) == 0:
        return np.empty((0, 4))
    return np.array(raw)[:, :4] * scale


def _bbox_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    areaB = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / float(areaA + areaB - inter)


def _join_bbox(tube, bbox):
    return (min(tube[0], bbox[0]), min(tube[1], bbox[1]),
            max(tube[2], bbox[2]), max(tube[3], bbox[3]))


def crop_video_to_25fps(input_path: str, out_path: str,
                        increase_area: float = 0.1, iou_threshold: float = 0.25):
    """Trajectory-based face crop (mirrors data_preprocess/crop_video.py).

    Tracks face bboxes across ALL frames, accumulates a tube_bbox per
    trajectory, then crops with aspect-preserving padding.
    """
    print("Detecting faces across all frames for crop (trajectory-based)...")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=get_fa_device(),
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    frame_shape = None
    # trajectory: [initial_bbox, tube_bbox, start_idx, end_idx]
    active: list = []
    completed: list = []

    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_shape is None:
            frame_shape = frame_rgb.shape

        bboxes = _detect_bboxes(frame_rgb, fa)

        # Expire trajectories where no current bbox overlaps the initial bbox
        still_active, lost = [], []
        for traj in active:
            best = max((_bbox_iou(traj[0], b) for b in bboxes), default=0.0)
            (still_active if best > iou_threshold else lost).append(traj)
        completed.extend(lost)
        active = still_active

        # Assign each detected bbox to the best-matching active trajectory
        for bbox in bboxes:
            best_score, best_traj = 0.0, None
            for traj in active:
                score = _bbox_iou(traj[0], bbox)
                if score > best_score and score > iou_threshold:
                    best_score, best_traj = score, traj
            if best_traj is None:
                active.append([bbox.copy(), bbox.copy(), idx, idx])
            else:
                best_traj[3] = idx
                best_traj[1] = _join_bbox(best_traj[1], bbox)

        idx += 1

    cap.release()
    completed.extend(active)

    if not completed:
        raise RuntimeError(
            "No face detected in any frame. "
            "Make sure your face is visible in the video."
        )

    # Longest trajectory → most stable face region
    _, tube_bbox, _, _ = max(completed, key=lambda t: t[3] - t[2])
    h, w = frame_shape[:2]
    left, top, right, bot = tube_bbox
    bw, bh = right - left, bot - top

    # Aspect-preserving expansion (same formula as compute_bbox in crop_video.py)
    w_inc = max(increase_area, ((1 + 2 * increase_area) * bh - bw) / (2 * bw))
    h_inc = max(increase_area, ((1 + 2 * increase_area) * bw - bh) / (2 * bh))
    left  = max(0, int(left  - w_inc * bw))
    top   = max(0, int(top   - h_inc * bh))
    right = min(w, int(right + w_inc * bw))
    bot   = min(h, int(bot   + h_inc * bh))
    cw, ch = right - left, bot - top

    cmd = (
        f'ffmpeg -i "{input_path}" '
        f'-vf "crop={cw}:{ch}:{left}:{top},scale=256:256" '
        f'-r 25 "{out_path}" -y -loglevel error'
    )
    print("Cropping video to 25 fps / 256x256...")
    ret = os.system(cmd)
    if ret != 0 or not os.path.exists(out_path):
        raise RuntimeError("ffmpeg crop failed. Make sure ffmpeg is installed.")


def extract_identity_frame(cropped_video_path: str, out_path: str):
    cap = cv2.VideoCapture(cropped_video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read cropped video: {cropped_video_path}")
    cv2.imwrite(out_path, frame)


def _write_silent_wav(out_path: str, sr: int = 16000, duration: float = 60.0):
    import wave
    num_frames = int(sr * duration)
    with wave.open(out_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00" * num_frames * 2)


def extract_audio(input_path: str, out_path: str):
    cmd = (
        f'ffmpeg -i "{input_path}" -f wav -acodec pcm_s16le -ar 16000 '
        f'"{out_path}" -y -loglevel error'
    )
    print("Extracting audio...")
    ret = os.system(cmd)
    if ret != 0 or not os.path.exists(out_path):
        print("[WARN] No audio stream found in video; using silent audio. "
              "Stage 2에서 Audio Source를 직접 선택하세요.")
        _write_silent_wav(out_path)


def preprocess(input_video: str, tmp_dir: str):
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    cropped_video = os.path.join(tmp_dir, "cropped.mp4")
    identity_img  = os.path.join(tmp_dir, "identity.png")
    audio_wav     = os.path.join(tmp_dir, "audio.wav")

    crop_video_to_25fps(input_video, cropped_video)
    extract_identity_frame(cropped_video, identity_img)
    extract_audio(input_video, audio_wav)

    return identity_img, audio_wav, cropped_video


def apply_super_resolution(video_path: str) -> str:
    """Apply GFPGAN face SR (256 → 512). Returns the enhanced video path."""
    if not check_package_installed("gfpgan"):
        return video_path

    # basicsr.data.degradations imports from torchvision.transforms.functional_tensor
    # which was removed in torchvision >= 0.16. Register a shim before gfpgan loads.
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        import types
        import torchvision.transforms.functional as _tvf
        _shim = types.ModuleType("torchvision.transforms.functional_tensor")
        for _attr in dir(_tvf):
            if not _attr.startswith("__"):
                setattr(_shim, _attr, getattr(_tvf, _attr))
        sys.modules["torchvision.transforms.functional_tensor"] = _shim

    from src.EDTalk.face_sr.face_enhancer import enhancer_list

    out_path = video_path.replace(".mp4", "_512.mp4")
    tmp_path = out_path + ".tmp.mp4"
    print("Applying super resolution (GFPGAN)...")
    imageio.mimsave(
        tmp_path,
        enhancer_list(video_path, method="gfpgan", bg_upsampler=None),
        fps=float(25),
        codec="libx264",
    )
    video_clip = VideoFileClip(tmp_path)
    audio_clip = AudioFileClip(video_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
    os.remove(tmp_path)
    return out_path


def create_comparison_video(input_video: str, result_video: str, emotion_name: str) -> str:
    """Create side-by-side comparison: input (left) | result (right) with text labels."""
    comparison_path = result_video.replace(".mp4", "_comparison.mp4")
    tmp_path = comparison_path + ".tmp.mp4"

    cap_in  = cv2.VideoCapture(input_video)
    cap_out = cv2.VideoCapture(result_video)

    frames = []
    while True:
        ret_in,  frame_in  = cap_in.read()
        ret_out, frame_out = cap_out.read()
        if not ret_in or not ret_out:
            break

        h, w = frame_out.shape[:2]
        frame_in = cv2.resize(frame_in, (w, h))

        font       = cv2.FONT_HERSHEY_DUPLEX
        scale      = w / 512 * 0.8
        thickness  = max(1, int(w / 256))
        white      = (255, 255, 255)
        black      = (0, 0, 0)
        y_pos      = max(30, int(h * 0.07))

        for text, frame in [("Input Video", frame_in), (emotion_name, frame_out)]:
            cv2.putText(frame, text, (10, y_pos), font, scale, black, thickness + 2)
            cv2.putText(frame, text, (10, y_pos), font, scale, white, thickness)

        combined = np.concatenate(
            [cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB),
             cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)],
            axis=1,
        )
        frames.append(combined)

    cap_in.release()
    cap_out.release()

    if not frames:
        return result_video

    imageio.mimsave(tmp_path, frames, fps=float(25))
    audio_clip = AudioFileClip(result_video)
    video_clip = VideoFileClip(tmp_path)
    video_clip.set_audio(audio_clip).write_videofile(
        comparison_path, codec="libx264", audio_codec="aac", logger=None
    )
    os.remove(tmp_path)
    return comparison_path


def create_multi_comparison_video(
    input_video: str,
    result_paths: list,
    emotion_names: list,
) -> str:
    """Horizontal N+1 panel: Input | emotion1 | emotion2 | ..."""
    save_path = "res/demo_multi_comparison.mp4"
    tmp_path  = save_path + ".tmp.mp4"

    caps   = [cv2.VideoCapture(input_video)] + [cv2.VideoCapture(p) for p in result_paths]
    labels = ["Input Video"] + list(emotion_names)

    frames = []
    while True:
        panel_frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                panel_frames = []
                break
            panel_frames.append(frame)
        if not panel_frames:
            break

        h_ref, w_ref = panel_frames[1].shape[:2]
        resized = [cv2.resize(f, (w_ref, h_ref)) for f in panel_frames]

        font      = cv2.FONT_HERSHEY_DUPLEX
        scale     = w_ref / 512 * 0.8
        thickness = max(1, int(w_ref / 256))
        y_pos     = max(30, int(h_ref * 0.07))

        for frame, label in zip(resized, labels):
            cv2.putText(frame, label, (10, y_pos), font, scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, label, (10, y_pos), font, scale, (255, 255, 255), thickness)

        combined_rgb = np.concatenate(
            [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in resized],
            axis=1,
        )
        frames.append(combined_rgb)

    for cap in caps:
        cap.release()

    if not frames:
        return result_paths[0]

    os.makedirs("res", exist_ok=True)
    imageio.mimsave(tmp_path, frames, fps=float(25))
    audio_clip = AudioFileClip(result_paths[0])
    video_clip = VideoFileClip(tmp_path)
    video_clip.set_audio(audio_clip).write_videofile(
        save_path, codec="libx264", audio_codec="aac", logger=None
    )
    os.remove(tmp_path)
    return save_path


def load_e2v_direction(neu_path: str, emo_path: str, num_samples: int = 10):
    neu_files = [
        os.path.join(neu_path, f) for f in os.listdir(neu_path) if f.endswith(".npy")
    ]
    emo_files = [
        os.path.join(emo_path, f) for f in os.listdir(emo_path) if f.endswith(".npy")
    ]

    if len(neu_files) < num_samples:
        raise RuntimeError(
            f"Not enough neutral e2v files (need {num_samples}, got {len(neu_files)})"
        )
    if len(emo_files) < num_samples:
        raise RuntimeError(
            f"Not enough emotion e2v files (need {num_samples}, got {len(emo_files)})"
        )

    neu = torch.stack(
        [torch.from_numpy(np.load(f)).float() for f in random.sample(neu_files, num_samples)]
    )
    emo = torch.stack(
        [torch.from_numpy(np.load(f)).float() for f in random.sample(emo_files, num_samples)]
    )
    return emo.mean(dim=0) - neu.mean(dim=0)


# ---------------------------------------------------------------------------
# Core inference — uses _MODELS global; no model loading inside
# ---------------------------------------------------------------------------

def run_single_inference(
    identity_img: str,
    audio_wav: str,
    cropped_video: str,
    emo_e2v_path: str,
    save_path: str,
    emotion_name: str = "",
    use_sr: bool = True,
    pose_video: str = None,
    num_samples: int = 10,
):
    device        = _MODELS["device"]
    T             = _MODELS["T"]
    audio2lip     = _MODELS["audio2lip"]
    gen           = _MODELS["gen"]
    connector_exp = _MODELS["connector_exp"]

    e2v = load_e2v_direction(NEU_E2V_PATH, emo_e2v_path, num_samples)
    e2v = e2v.unsqueeze(0).unsqueeze(0).to(device)

    img_source = img_preprocessing(identity_img, 256).to(device)

    audio, audio_bs, audio_T = audio_preprocessing(audio_wav, device=device)
    lip_vid_target = audio2lip(audio, audio_bs, audio_T)[0]
    lip_vid_target = conv_feat(lip_vid_target, k_size=3, sigma=1).to(device)
    lip_len = lip_vid_target.size(0)

    _pose_src = pose_video or cropped_video
    pose_vid_target, fps = vid_preprocessing(_pose_src)
    pose_vid_target = pose_vid_target.to(device)
    len_pose = pose_vid_target.shape[1]

    src_vid_target, _ = vid_preprocessing(cropped_video)
    src_vid_target = src_vid_target.to(device)
    vid_len = src_vid_target.shape[1] - src_vid_target.shape[1] % T

    ED_ref_T = torch.zeros((1, T, 10)).to(device)
    predicted_alpha_D_exp = []
    with torch.no_grad():
        batch_vid = src_vid_target.view(
            -1,
            src_vid_target.size(2),
            src_vid_target.size(3),
            src_vid_target.size(4),
        )
        ED_neu, _, _ = gen.compute_alpha_D(batch_vid)
        ED_neu = ED_neu.unsqueeze(0).to(device)

        for i in range(0, vid_len, T):
            ED_neu_T = ED_neu[:, i:i + T, :]
            pred_exp_dir, _ = connector_exp(ED_ref_T, e2v, ED_neu_T)
            pred_exp = ED_neu_T.squeeze(0) + pred_exp_dir
            ED_ref_T = pred_exp_dir.unsqueeze(0)
            predicted_alpha_D_exp.append(pred_exp)

    exp_vid_target = torch.cat(predicted_alpha_D_exp, dim=0).unsqueeze(0)
    exp_vid_target = exp_vid_target[:, :-20]
    while exp_vid_target.shape[1] < lip_len:
        exp_vid_target = torch.cat(
            [exp_vid_target, torch.flip(exp_vid_target, dims=[1])], dim=1
        )
    exp_vid_target = exp_vid_target[:lip_len]
    exp_len = exp_vid_target.shape[1]

    vid_target_recon = []
    with torch.no_grad():
        for i in tqdm(range(lip_len)):
            img_target_lip  = lip_vid_target[i:i + 1]
            img_target_pose = pose_vid_target[:, min(i, len_pose - 1), :, :, :]
            alpha_D_exp     = exp_vid_target[:, min(i, exp_len - 1), :]
            img_recon = gen.test_EDTalk_AV_use_exp_weight(
                img_source, img_target_lip, img_target_pose, alpha_D_exp, h_start=None
            )
            vid_target_recon.append(img_recon.unsqueeze(2))

    vid_target_recon = torch.cat(vid_target_recon, dim=2)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    temp_path = save_path.replace(".mp4", "_temp.mp4")
    save_video(vid_target_recon, temp_path, fps)
    cmd = (
        f'ffmpeg -y -i "{temp_path}" -i "{audio_wav}" '
        f'-vcodec copy "{save_path}" -loglevel error'
    )
    os.system(cmd)
    os.remove(temp_path)

    sr_path = apply_super_resolution(save_path) if use_sr else save_path
    comparison_path = create_comparison_video(cropped_video, sr_path, emotion_name)
    return sr_path, comparison_path


# ---------------------------------------------------------------------------
# Gradio-specific helpers
# ---------------------------------------------------------------------------

def get_audio_samples(display_name: str) -> list:
    """Return all .wav sample paths for the given emotion display name."""
    if not display_name or display_name not in EMOTION_DISPLAY_MAP:
        return []
    _, _, e2v_path = EMOTION_DISPLAY_MAP[display_name]
    audio_dir = os.path.dirname(e2v_path)
    wavs = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    return [os.path.join(audio_dir, f) for f in wavs]


def auto_transcribe_gradio(video_path):
    """Extract audio from video and transcribe with Whisper. Used as a .then() chain."""
    if not video_path:
        return gr.update(), gr.update()

    os.makedirs(TMP_VC_DIR, exist_ok=True)
    source_audio = os.path.join(TMP_VC_DIR, "source.wav")
    extract_audio(video_path, source_audio)

    try:
        text = transcribe_audio(source_audio)
    except Exception as e:
        print(f"[WARN] Transcription failed: {e}")
        text = ""

    status = "✅ 영상이 설정되었습니다. 전사 완료. Stage 2로 이동하세요." if text else \
             "✅ 영상이 설정되었습니다. (전사 실패 — 텍스트를 직접 입력하세요.)"
    return gr.update(value=text), gr.update(value=status)


def generate_voice_gradio(
    video_path,
    transcribed_text,
    selected_display_names,
    emotion_strength,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate emotion-controlled voice clones for each selected emotion."""
    if not video_path:
        raise gr.Error("Stage 1에서 영상을 먼저 설정해주세요.")
    if not selected_display_names:
        raise gr.Error("감정을 하나 이상 선택해주세요.")
    if not transcribed_text or not transcribed_text.strip():
        raise gr.Error("전사된 텍스트가 없습니다. 텍스트를 직접 입력해주세요.")

    os.makedirs(TMP_VC_DIR, exist_ok=True)
    source_audio = os.path.join(TMP_VC_DIR, "source.wav")
    if not os.path.exists(source_audio):
        extract_audio(video_path, source_audio)

    progress(0.05, desc="MetaVoice 모델 로딩 중 (첫 실행 시 다소 시간이 걸립니다)...")
    try:
        load_voice_cloning_models(output_dir=TMP_VC_DIR)
    except RuntimeError as e:
        raise gr.Error(str(e))

    voice_results = {}  # display_name → cloned audio path (no "원본" key)
    total = len(selected_display_names)

    for i, display_name in enumerate(selected_display_names):
        progress(
            (i + 0.1) / (total + 0.1),
            desc=f"🎙 {display_name} 음성 생성 중... ({i + 1}/{total})",
        )
        safe = display_name.replace(" ", "_").replace("[", "").replace("]", "")
        out_path = os.path.join(TMP_VC_DIR, f"voice_{safe}.wav")
        try:
            clone_voice_with_emotion(
                text=transcribed_text.strip(),
                source_audio=source_audio,
                emotion_display_name=display_name,
                strength=emotion_strength,
                output_path=out_path,
            )
            voice_results[display_name] = out_path
        except Exception as e:
            print(f"[WARN] Voice cloning failed for {display_name}: {e}")

    # Build dropdown choices: original first, then each cloned emotion
    choices = ["원본 (Original)"] + list(voice_results.keys())
    display_map = {"원본 (Original)": source_audio, **voice_results}

    progress(1.0, desc="완료!")

    return (
        gr.update(choices=choices, value="원본 (Original)", visible=True),
        gr.update(value=source_audio, visible=True),
        voice_results,  # only emotion keys → stored in voice_results_state
        display_map,    # includes "원본" → stored in voice_display_state for dropdown
    )


def run_inference_gradio(
    video_path,
    selected_display_names,
    use_sr: bool = True,
    identity_asset_dd: str = _SRC_DEFAULT,
    identity_upload=None,
    pose_asset_dd: str = _SRC_DEFAULT,
    pose_upload=None,
    audio_asset_dd: str = _SRC_DEFAULT,
    audio_upload=None,
    voice_results: dict = None,
    progress=gr.Progress(track_tqdm=True),
):
    """Preprocess once, then run inference per selected emotion. Returns first result."""
    if not video_path:
        raise gr.Error("Stage 1에서 영상을 먼저 설정해주세요.")
    if not selected_display_names:
        raise gr.Error("감정을 하나 이상 선택해주세요.")

    os.makedirs("res", exist_ok=True)
    progress(0, desc="전처리 중...")

    try:
        identity_img, audio_wav, cropped_video = preprocess(video_path, TMP_DIR)
    except RuntimeError as e:
        raise gr.Error(f"전처리 실패: {e}")

    def _resolve(upload, dropdown, asset_map):
        if upload:
            return upload
        if dropdown and dropdown != _SRC_DEFAULT:
            return asset_map.get(dropdown)
        return None

    identity_src = _resolve(identity_upload, identity_asset_dd, SAMPLE_IDENTITY) or identity_img
    audio_src    = _resolve(audio_upload, audio_asset_dd, SAMPLE_AUDIO) or audio_wav

    pose_override = _resolve(pose_upload, pose_asset_dd, SAMPLE_VIDEOS)
    if pose_override and pose_override not in set(SAMPLE_VIDEOS.values()):
        tmp_pose = os.path.join(TMP_DIR, "pose_override_cropped.mp4")
        crop_video_to_25fps(pose_override, tmp_pose)
        pose_override = tmp_pose
    pose_src = pose_override or cropped_video

    results = {}  # display_name -> (sr_path, comparison_path)
    total = len(selected_display_names)

    try:
        for i, display_name in enumerate(selected_display_names):
            progress(
                (i + 0.1) / total,
                desc=f"{display_name} 추론 중... ({i + 1}/{total})",
            )
            _, _, emo_e2v_path = EMOTION_DISPLAY_MAP[display_name]
            safe = display_name.replace(" ", "_").replace("[", "").replace("]", "")
            save_path = f"res/demo_{safe}.mp4"

            # Use per-emotion voice-cloned audio if available; fall back to audio_src
            if voice_results and display_name in voice_results:
                emotion_audio = voice_results[display_name]
            else:
                emotion_audio = audio_src

            sr_path, comparison_path = run_single_inference(
                identity_src, emotion_audio, cropped_video, emo_e2v_path, save_path,
                emotion_name=display_name,
                use_sr=use_sr,
                pose_video=pose_src,
            )
            results[display_name] = (sr_path, comparison_path)

        if len(results) > 1:
            sr_paths = [results[n][0] for n in selected_display_names]
            combined_cmp = create_multi_comparison_video(
                cropped_video, sr_paths, selected_display_names
            )
        else:
            combined_cmp = results[selected_display_names[0]][1]
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    progress(1.0, desc="완료!")

    first_display = selected_display_names[0]
    first_sr, first_cmp = results[first_display]
    choices = list(results.keys())
    multi = len(choices) > 1

    return (
        gr.update(value=first_sr, visible=True),
        gr.update(value=first_cmp, visible=True),
        gr.update(choices=choices, value=first_display, visible=multi),
        results,
        gr.update(value=combined_cmp, visible=multi),
    )


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="C-MET Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# C-MET — Cross-Modal Emotion Transfer\n"
            "웹캠으로 영상을 촬영하고, 원하는 감정이 담긴 토킹 페이스 영상을 생성합니다."
        )

        # Shared state
        video_state        = gr.State(None)
        results_state      = gr.State({})
        voice_results_state = gr.State({})   # display_name → cloned audio path
        voice_display_state = gr.State({})   # includes "원본" key for dropdown

        with gr.Tabs() as tabs:

            # ── Stage 1: Record ───────────────────────────────────────
            with gr.Tab("① Record Video", id="tab_record"):
                gr.Markdown(
                    "**웹캠 촬영** 또는 **비디오 파일 업로드** 중 하나를 선택하세요.\n\n"
                    "입력 완료 후 **Use This Video →** 버튼을 클릭하면 Stage 2로 이동합니다."
                )
                with gr.Row():
                    btn_webcam_mode = gr.Button(
                        "📷 웹캠으로 촬영", variant="secondary", size="sm"
                    )
                    btn_upload_mode = gr.Button(
                        "📁 비디오 파일 업로드", variant="primary", size="sm"
                    )
                webcam_input = gr.Video(
                    sources=["webcam"],
                    include_audio=True,
                    label="웹캠 촬영",
                    height=400,
                    visible=False,
                )
                upload_input = gr.Video(
                    sources=["upload"],
                    label="비디오 파일 업로드",
                    height=400,
                    visible=True,
                )
                use_video_btn = gr.Button(
                    "Use This Video →", variant="primary", size="lg"
                )
                video_status = gr.Markdown("")

                if SAMPLE_VIDEOS:
                    gr.Markdown("---\n#### 또는 샘플 비디오 사용")
                    sample_dropdown = gr.Dropdown(
                        choices=list(SAMPLE_VIDEOS.keys()),
                        label="샘플 영상 선택",
                        value=None,
                    )
                    sample_preview = gr.Video(
                        label="샘플 미리보기",
                        interactive=False,
                        visible=False,
                        height=300,
                    )
                    use_sample_btn = gr.Button(
                        "샘플 비디오 사용 →", variant="secondary", size="lg"
                    )

            # ── Stage 2: Emotions & Voice Cloning ────────────────────
            with gr.Tab("② Emotions & Voice", id="tab_emotion"):
                gr.Markdown(
                    "감정을 선택하고 오디오 샘플을 미리 들어보세요.  \n"
                    "그 다음 **감정 음성 생성**으로 입력 목소리에 감정을 입혀 확인하고 Stage 3으로 넘어가세요."
                )

                emotion_checkboxes = gr.CheckboxGroup(
                    choices=EMOTION_DISPLAY_NAMES,
                    label="감정 선택 (중복 가능)",
                )
                selected_summary = gr.Markdown("선택된 감정: **없음**")

                gr.Markdown("---\n#### 오디오 샘플 미리듣기")
                preview_dropdown = gr.Dropdown(
                    choices=EMOTION_DISPLAY_NAMES,
                    label="미리들을 감정",
                )
                sample_choice_dropdown = gr.Dropdown(
                    choices=[],
                    label="샘플 선택",
                    visible=False,
                    interactive=True,
                )
                preview_btn = gr.Button("▶ Play", visible=False)
                audio_preview = gr.Audio(
                    label="Emotion Sample",
                    autoplay=True,
                    visible=False,
                    interactive=False,
                )

                gr.Markdown("---\n#### 감정 음성 생성 (Voice Cloning)")
                transcription_box = gr.Textbox(
                    label="전사된 텍스트 (자동 입력 · 편집 가능)",
                    placeholder="Stage 1에서 영상을 설정하면 자동으로 전사됩니다...",
                    lines=3,
                    interactive=True,
                )
                emotion_strength_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    label="감정 강도 (Emotion Strength)",
                )
                generate_voice_btn = gr.Button(
                    "🎙 Generate Emotional Voice", variant="primary"
                )

                gr.Markdown("##### 음성 생성 결과 미리듣기")
                voice_result_dropdown = gr.Dropdown(
                    choices=[],
                    label="결과 선택 (원본 / 감정별 클론)",
                    visible=False,
                    interactive=True,
                )
                voice_result_audio = gr.Audio(
                    label="음성 미리듣기",
                    autoplay=False,
                    visible=False,
                    interactive=False,
                )

                sr_checkbox = gr.Checkbox(
                    label="Super Resolution 적용 (256 → 512, GFPGAN)",
                    value=True,
                )

                with gr.Accordion("소스 설정 (선택 사항)", open=False):
                    gr.Markdown(
                        "기본값은 입력 비디오에서 자동으로 추출됩니다. "
                        "Voice Cloning을 사용하면 감정 음성이 lip sync 소스로 자동 적용됩니다."
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("**Identity (얼굴 이미지)**")
                            identity_asset_dd = gr.Dropdown(
                                choices=[_SRC_DEFAULT] + list(SAMPLE_IDENTITY.keys()),
                                value=_SRC_DEFAULT,
                                label="asset/identity 선택",
                            )
                            identity_upload = gr.Image(
                                type="filepath",
                                label="또는 이미지 업로드",
                            )
                        with gr.Column(scale=1):
                            gr.Markdown("**Pose Source (머리 포즈)**")
                            pose_asset_dd = gr.Dropdown(
                                choices=[_SRC_DEFAULT] + list(SAMPLE_VIDEOS.keys()),
                                value=_SRC_DEFAULT,
                                label="asset/video 선택",
                            )
                            pose_upload = gr.Video(
                                sources=["upload"],
                                label="또는 비디오 업로드",
                            )
                        with gr.Column(scale=1):
                            gr.Markdown("**Audio Source (립싱크 — Voice Cloning 미사용 시)**")
                            audio_asset_dd = gr.Dropdown(
                                choices=[_SRC_DEFAULT] + list(SAMPLE_AUDIO.keys()),
                                value=_SRC_DEFAULT,
                                label="asset/audio 선택",
                            )
                            audio_upload = gr.Audio(
                                type="filepath",
                                label="또는 오디오 업로드",
                            )

                with gr.Row():
                    back_to_record = gr.Button("← Back to Record")
                    run_btn = gr.Button("Run Inference →", variant="primary")

            # ── Stage 3: Results ──────────────────────────────────────
            with gr.Tab("③ Results", id="tab_result"):
                status_md = gr.Markdown(
                    "추론이 완료되면 결과가 여기에 표시됩니다."
                )
                result_selector = gr.Dropdown(
                    choices=[], label="결과 선택 (감정)", visible=False
                )
                result_video = gr.Video(
                    label="생성 결과 (SR 적용)",
                    visible=False,
                    height=400,
                )
                gr.Markdown("---\n#### 입력 vs 결과 비교")
                comparison_video = gr.Video(
                    label="비교 영상 (Input | 결과)",
                    visible=False,
                    height=400,
                )
                gr.Markdown("---\n#### 전체 감정 비교 (Input | emotion1 | emotion2 | ...)")
                multi_comparison_video = gr.Video(
                    label="전체 감정 비교 영상",
                    visible=False,
                    height=400,
                )
                back_to_emotion = gr.Button("← Select More Emotions")

        # ── Event wiring ─────────────────────────────────────────────

        btn_webcam_mode.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[webcam_input, upload_input],
        )
        btn_upload_mode.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[webcam_input, upload_input],
        )

        def on_use_video(webcam_path, upload_path):
            video_path = webcam_path or upload_path
            if video_path is None:
                return (
                    gr.update(value="⚠️ 먼저 영상을 촬영하거나 업로드해주세요."),
                    gr.update(),
                    None,
                )
            return (
                gr.update(value="✅ 영상이 설정되었습니다. Stage 2로 이동하세요."),
                gr.update(selected="tab_emotion"),
                video_path,
            )

        use_video_btn.click(
            on_use_video,
            inputs=[webcam_input, upload_input],
            outputs=[video_status, tabs, video_state],
        ).then(
            auto_transcribe_gradio,
            inputs=[video_state],
            outputs=[transcription_box, video_status],
        )

        if SAMPLE_VIDEOS:
            def on_sample_select(name):
                if not name:
                    return gr.update(visible=False)
                return gr.update(value=SAMPLE_VIDEOS[name], visible=True)

            sample_dropdown.change(
                on_sample_select,
                inputs=[sample_dropdown],
                outputs=[sample_preview],
            )

            def on_use_sample(name):
                if not name:
                    return (
                        gr.update(value="⚠️ 샘플 영상을 선택해주세요."),
                        gr.update(),
                        None,
                    )
                return (
                    gr.update(value=f"✅ 샘플 '{name}' 설정되었습니다. Stage 2로 이동하세요."),
                    gr.update(selected="tab_emotion"),
                    SAMPLE_VIDEOS[name],
                )

            use_sample_btn.click(
                on_use_sample,
                inputs=[sample_dropdown],
                outputs=[video_status, tabs, video_state],
            ).then(
                auto_transcribe_gradio,
                inputs=[video_state],
                outputs=[transcription_box, video_status],
            )

        def on_emotion_change(selected):
            if not selected:
                return "선택된 감정: **없음**"
            return f"선택된 감정: **{', '.join(selected)}** ({len(selected)}개)"

        emotion_checkboxes.change(
            on_emotion_change,
            inputs=[emotion_checkboxes],
            outputs=[selected_summary],
        )

        generate_voice_btn.click(
            fn=generate_voice_gradio,
            inputs=[
                video_state, transcription_box,
                emotion_checkboxes, emotion_strength_slider,
            ],
            outputs=[
                voice_result_dropdown, voice_result_audio,
                voice_results_state, voice_display_state,
            ],
        )

        def on_voice_result_change(selected, display_map):
            if not selected or not display_map or selected not in display_map:
                return gr.update()
            return gr.update(value=display_map[selected])

        voice_result_dropdown.change(
            on_voice_result_change,
            inputs=[voice_result_dropdown, voice_display_state],
            outputs=[voice_result_audio],
        )

        def on_emotion_preview_change(display_name):
            samples = get_audio_samples(display_name)
            if not samples:
                return (
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
            labels = [os.path.basename(p) for p in samples]
            path_map = dict(zip(labels, samples))
            first_label = labels[0]
            return (
                gr.update(choices=list(path_map.keys()), value=first_label, visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        preview_dropdown.change(
            on_emotion_preview_change,
            inputs=[preview_dropdown],
            outputs=[sample_choice_dropdown, preview_btn, audio_preview],
        )

        def on_preview(display_name, sample_label):
            if not display_name or not sample_label:
                return gr.update(visible=False)
            samples = get_audio_samples(display_name)
            labels = [os.path.basename(p) for p in samples]
            path_map = dict(zip(labels, samples))
            path = path_map.get(sample_label)
            if path is None:
                return gr.update(visible=False)
            return gr.update(value=path, visible=True)

        preview_btn.click(
            on_preview,
            inputs=[preview_dropdown, sample_choice_dropdown],
            outputs=[audio_preview],
        )

        back_to_record.click(
            lambda: gr.update(selected="tab_record"),
            outputs=[tabs],
        )

        back_to_emotion.click(
            lambda: gr.update(selected="tab_emotion"),
            outputs=[tabs],
        )

        # Run inference: immediately switch tab + update status, then infer
        run_btn.click(
            fn=lambda: (
                gr.update(selected="tab_result"),
                "⏳ 추론 중입니다. 잠시 기다려주세요...",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            outputs=[tabs, status_md, result_video, comparison_video, result_selector, multi_comparison_video],
        ).then(
            fn=run_inference_gradio,
            inputs=[
                video_state, emotion_checkboxes, sr_checkbox,
                identity_asset_dd, identity_upload,
                pose_asset_dd, pose_upload,
                audio_asset_dd, audio_upload,
                voice_results_state,
            ],
            outputs=[result_video, comparison_video, result_selector, results_state, multi_comparison_video],
        ).then(
            fn=lambda: "✅ 추론 완료! 아래에서 결과를 확인하세요.",
            outputs=[status_md],
        )

        # Switch both videos when dropdown changes
        def on_selector_change(name, res):
            if not name or name not in res:
                return gr.update(), gr.update()
            sr_path, cmp_path = res[name]
            return gr.update(value=sr_path), gr.update(value=cmp_path)

        result_selector.change(
            fn=on_selector_change,
            inputs=[result_selector, results_state],
            outputs=[result_video, comparison_video],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    fix_seed(42)
    load_models()
    app = build_gradio_app()
    app.queue()
    app.launch(inbrowser=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
