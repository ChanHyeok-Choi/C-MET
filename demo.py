import os
import sys
import random
import shutil

# MPS 미지원 연산(torch.qr 등)을 CPU로 자동 폴백 — torch import 전에 설정해야 적용됨
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _patch_gradio_client():
    """
    gradio_client 1.3.x 버그 패치: JSON Schema에서 additionalProperties가
    불리언(true/false)일 때 get_type()과 _json_schema_to_python_type()이
    TypeError/AttributeError를 발생시키는 문제를 수정합니다.
    """
    try:
        import gradio_client.utils as _u

        _orig_get_type = _u.get_type

        def _get_type(schema):
            if not isinstance(schema, dict):
                return {}
            return _orig_get_type(schema)

        _u.get_type = _get_type

        _orig_convert = _u._json_schema_to_python_type

        def _json_schema_to_python_type(schema, defs):
            if isinstance(schema, bool):
                return "Any"
            return _orig_convert(schema, defs)

        _u._json_schema_to_python_type = _json_schema_to_python_type
    except Exception:
        pass


_patch_gradio_client()

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

# ---------------------------------------------------------------------------
# Emotion catalogue
# ---------------------------------------------------------------------------

EMOTIONS = [
    ("neutral",     "MEAD",   "audios/MEAD/neutral/emotion2vec+large_features"),
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

TMP_DIR = "tmp_demo"

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

_SRC_DEFAULT = "선택 안 함"

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
    w = torch.load(config.audio2lip_model_path, weights_only=False, map_location=lambda s, l: s)["audio2lip"]
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
    w = torch.load(pretrained_EDTalk["model_path"], weights_only=False, map_location=lambda s, l: s)["gen"]
    gen.load_state_dict(w)
    gen.eval()

    connector_exp = Connector_exp(projector_kwargs, transformer_kwargs, device).to(device)
    w = torch.load(config.connector_exp_path, weights_only=False, map_location=lambda s, l: s)
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


def crop_image_to_face(input_path: str, out_path: str, increase_area: float = 0.1):
    """Face-crop a single image using the same aspect-preserving padding as
    crop_video_to_25fps, but detecting on a single frame instead of a trajectory."""
    frame_bgr = cv2.imread(input_path)
    if frame_bgr is None:
        raise RuntimeError(f"Cannot read image: {input_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=get_fa_device(),
    )
    bboxes = _detect_bboxes(frame_rgb, fa)
    if len(bboxes) == 0:
        raise RuntimeError("이미지에서 얼굴을 감지할 수 없습니다. 다른 이미지를 사용해주세요.")

    left, top, right, bot = bboxes[0]
    bw, bh = right - left, bot - top
    h, w = frame_rgb.shape[:2]
    w_inc = max(increase_area, ((1 + 2 * increase_area) * bh - bw) / (2 * bw))
    h_inc = max(increase_area, ((1 + 2 * increase_area) * bw - bh) / (2 * bh))
    left  = max(0, int(left  - w_inc * bw))
    top   = max(0, int(top   - h_inc * bh))
    right = min(w, int(right + w_inc * bw))
    bot   = min(h, int(bot   + h_inc * bh))

    cropped = cv2.resize(frame_bgr[top:bot, left:right], (256, 256))
    cv2.imwrite(out_path, cropped)


def prepare_combo_identity_video(
    identity_img_path: str,
    cropped_identity_img_path: str,
    out_video_path: str,
    num_frames: int = 50,
    fps: int = 25,
):
    """Face-crop the identity image and repeat it into a short pseudo-video.

    run_single_inference needs a multi-frame 'cropped_video' to compute the
    ED_neu baseline via gen.compute_alpha_D — that call batches per-frame with
    no temporal coupling, so repeating one identity frame T times yields a
    constant, identity-only neutral baseline instead of borrowing one from a
    separate pose-actor video.
    """
    crop_image_to_face(identity_img_path, cropped_identity_img_path)
    frame_bgr = cv2.imread(cropped_identity_img_path)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    imageio.mimsave(out_video_path, [frame_rgb] * num_frames, fps=float(fps))


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


def build_running_merge(entries: list) -> str:
    """Horizontal panel of every result generated so far this session.

    entries: [{"label": str, "sr_path": str}, ...] in generation order.
    No Input panel — durations/identities can differ across separate Run
    clicks, so there's no single coherent "input" to anchor it to. Always
    rebuilt from scratch and written to a fixed path.
    """
    save_path = "res/demo_running_merge.mp4"
    tmp_path  = save_path + ".tmp.mp4"

    caps   = [cv2.VideoCapture(e["sr_path"]) for e in entries]
    labels = [e["label"] for e in entries]

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

        h_ref, w_ref = panel_frames[0].shape[:2]
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
        return entries[0]["sr_path"]

    os.makedirs("res", exist_ok=True)
    imageio.mimsave(tmp_path, frames, fps=float(25))
    audio_clip = AudioFileClip(entries[0]["sr_path"])
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
    # Ping-pong pad if the pose clip is shorter than the audio-driven length
    # (e.g. combo mode lets pose/audio durations differ independently) —
    # otherwise indexing below would clamp to the last frame and freeze.
    while pose_vid_target.shape[1] < lip_len:
        pose_vid_target = torch.cat(
            [pose_vid_target, torch.flip(pose_vid_target, dims=[1])], dim=1
        )
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


def _preprocess_combo_input(input_state: dict, tmp_dir: str):
    """Combo mode: identity image + audio + pose video, no main video.

    cropped_video is replaced by a short repeated-frame pseudo-video built
    from the identity image (see prepare_combo_identity_video) so the
    ED_neu baseline reflects the identity, not a separate pose actor.
    """
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    identity_img    = os.path.join(tmp_dir, "identity.png")
    identity_pseudo = os.path.join(tmp_dir, "identity_pseudo.mp4")
    pose_cropped    = os.path.join(tmp_dir, "pose_cropped.mp4")

    prepare_combo_identity_video(input_state["identity"], identity_img, identity_pseudo)
    crop_video_to_25fps(input_state["pose"], pose_cropped)

    return identity_img, input_state["audio"], identity_pseudo, pose_cropped


def _resolve_asset(upload, dropdown, asset_map):
    if upload:
        return upload
    if dropdown and dropdown != _SRC_DEFAULT:
        return asset_map.get(dropdown)
    return None


def resolve_input_state(
    mode, webcam_path, upload_path, sample_name,
    identity_dd, identity_up, audio_dd, audio_up, pose_dd, pose_up,
) -> dict:
    """Resolve the currently-visible input widgets into an input_state dict.

    Raises gr.Error with a Korean message if the active mode's required
    pieces aren't all present — called fresh on every Run click instead of
    via a separate staging step.
    """
    if mode == "비디오로 입력":
        video_path = webcam_path or upload_path or (
            SAMPLE_VIDEOS.get(sample_name) if sample_name else None
        )
        if not video_path:
            raise gr.Error("영상을 촬영, 업로드, 또는 샘플에서 선택해주세요.")
        return {"mode": "video", "video": video_path}

    identity_src = _resolve_asset(identity_up, identity_dd, SAMPLE_IDENTITY)
    audio_src    = _resolve_asset(audio_up, audio_dd, SAMPLE_AUDIO)
    pose_src     = _resolve_asset(pose_up, pose_dd, SAMPLE_VIDEOS)
    if not (identity_src and audio_src and pose_src):
        raise gr.Error("Identity 이미지, Audio, Pose 영상을 모두 설정해주세요.")
    return {"mode": "combo", "identity": identity_src, "audio": audio_src, "pose": pose_src}


def run_inference_single_page(
    mode, webcam_path, upload_path, sample_name,
    identity_dd, identity_up, audio_dd, audio_up, pose_dd, pose_up,
    selected_display_names, use_sr: bool,
    results_so_far: list,
    progress=gr.Progress(track_tqdm=True),
):
    """Resolve input, run inference per selected emotion, and append the new
    results onto results_so_far (never replacing earlier Run clicks' results)."""
    if not selected_display_names:
        raise gr.Error("감정을 하나 이상 선택해주세요.")

    input_state = resolve_input_state(
        mode, webcam_path, upload_path, sample_name,
        identity_dd, identity_up, audio_dd, audio_up, pose_dd, pose_up,
    )

    os.makedirs("res", exist_ok=True)
    progress(0, desc="전처리 중...")

    try:
        if input_state["mode"] == "video":
            identity_img, audio_src, cropped_video = preprocess(input_state["video"], TMP_DIR)
            pose_src = cropped_video
        else:
            identity_img, audio_src, cropped_video, pose_src = _preprocess_combo_input(
                input_state, TMP_DIR
            )
    except RuntimeError as e:
        raise gr.Error(f"전처리 실패: {e}")

    base_idx = len(results_so_far)
    new_entries = []
    total = len(selected_display_names)

    try:
        for i, display_name in enumerate(selected_display_names):
            progress(
                (i + 0.1) / total,
                desc=f"{display_name} 추론 중... ({i + 1}/{total})",
            )
            _, _, emo_e2v_path = EMOTION_DISPLAY_MAP[display_name]
            run_idx = base_idx + i
            safe = display_name.replace(" ", "_").replace("[", "").replace("]", "")
            # run_idx keeps filenames unique across separate Run clicks so an
            # earlier stacked block's video file is never silently overwritten.
            save_path = f"res/demo_{run_idx:03d}_{safe}.mp4"

            sr_path, comparison_path = run_single_inference(
                identity_img, audio_src, cropped_video, emo_e2v_path, save_path,
                emotion_name=display_name,
                use_sr=use_sr,
                pose_video=pose_src,
            )
            new_entries.append({
                "label": f"#{run_idx + 1} · {display_name}",
                "sr_path": sr_path,
                "comparison_path": comparison_path,
            })
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    progress(0.9, desc="전체 결과 병합 중...")
    updated_results = results_so_far + new_entries
    merged_path = build_running_merge(updated_results)
    progress(1.0, desc="완료!")

    return (
        updated_results,
        gr.update(value=merged_path, visible=True),
        "✅ 추론 완료! 아래에서 결과를 확인하세요.",
    )


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="C-MET Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# C-MET — Cross-Modal Emotion Transfer\n"
            "입력을 설정하고 감정을 선택한 뒤 **Run Inference**를 누르면, "
            "결과가 아래에 계속 쌓이고 맨 아래에는 지금까지 생성된 모든 결과가 나란히 합쳐져 보입니다."
        )

        # Accumulates across every Run click this session: [{"label", "sr_path", "comparison_path"}, ...]
        results_list_state = gr.State([])

        # ── Input ──────────────────────────────────────────────────────
        input_mode_radio = gr.Radio(
            choices=["비디오로 입력", "Identity + Audio + Pose 조합"],
            value="비디오로 입력",
            label="입력 방식",
        )

        with gr.Group(visible=True) as video_mode_group:
            gr.Markdown("**웹캠 촬영** 또는 **비디오 파일 업로드** 중 하나를 선택하세요.")
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

            if SAMPLE_VIDEOS:
                gr.Markdown("---\n#### 또는 샘플 비디오 사용")
                sample_dropdown = gr.Dropdown(
                    choices=[_SRC_DEFAULT] + list(SAMPLE_VIDEOS.keys()),
                    value=_SRC_DEFAULT,
                    label="샘플 영상 선택",
                )
                sample_preview = gr.Video(
                    label="샘플 미리보기",
                    interactive=False,
                    visible=False,
                    height=300,
                )

        with gr.Group(visible=False) as combo_mode_group:
            gr.Markdown("**Identity 이미지**, **Audio**, **Pose 영상**을 각각 설정하세요.")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Identity (얼굴 이미지)**")
                    identity_asset_dd = gr.Dropdown(
                        choices=[_SRC_DEFAULT] + list(SAMPLE_IDENTITY.keys()),
                        value=_SRC_DEFAULT,
                        label="asset/identity 선택",
                    )
                    identity_asset_preview = gr.Image(
                        label="샘플 미리보기",
                        interactive=False,
                        visible=False,
                        height=200,
                    )
                    identity_upload = gr.Image(
                        type="filepath",
                        label="또는 이미지 업로드",
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**Audio (립싱크 소스)**")
                    audio_asset_dd = gr.Dropdown(
                        choices=[_SRC_DEFAULT] + list(SAMPLE_AUDIO.keys()),
                        value=_SRC_DEFAULT,
                        label="asset/audio 선택",
                    )
                    audio_asset_preview = gr.Audio(
                        label="샘플 미리보기",
                        interactive=False,
                        visible=False,
                    )
                    audio_upload = gr.Audio(
                        type="filepath",
                        label="또는 오디오 업로드",
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**Pose 영상 (머리 포즈)**")
                    pose_asset_dd = gr.Dropdown(
                        choices=[_SRC_DEFAULT] + list(SAMPLE_VIDEOS.keys()),
                        value=_SRC_DEFAULT,
                        label="asset/video 선택",
                    )
                    pose_asset_preview = gr.Video(
                        label="샘플 미리보기",
                        interactive=False,
                        visible=False,
                        height=200,
                    )
                    pose_upload = gr.Video(
                        sources=["upload"],
                        label="또는 비디오 업로드",
                    )

        # ── Emotions ──────────────────────────────────────────────────
        gr.Markdown("---\n## 감정 선택")
        emotion_checkboxes = gr.CheckboxGroup(
            choices=EMOTION_DISPLAY_NAMES,
            label="얼굴 표정을 바꿀 감정 (중복 가능)",
        )
        selected_summary = gr.Markdown("선택된 감정: **없음**")

        with gr.Accordion("오디오 샘플 미리듣기", open=False):
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

        sr_checkbox = gr.Checkbox(
            label="Super Resolution 적용 (256 → 512, GFPGAN)",
            value=True,
        )

        run_btn = gr.Button("Run Inference ↓", variant="primary", size="lg")
        status_md = gr.Markdown("")

        # ── Stacked results (grows on every Run click) ──────────────────
        gr.Markdown("---\n## 생성 결과")

        @gr.render(inputs=[results_list_state])
        def render_results(results):
            if not results:
                gr.Markdown("_아직 생성된 결과가 없습니다._")
                return
            for entry in results:
                gr.Markdown(f"#### {entry['label']}")
                with gr.Row():
                    gr.Video(value=entry["sr_path"], label="생성 결과 (SR 적용)", height=300)
                    gr.Video(value=entry["comparison_path"], label="Input vs 결과", height=300)

        gr.Markdown("---\n## 전체 병합 (지금까지 생성된 모든 결과, 새로고침 전까지 누적)")
        running_merge_video = gr.Video(
            label="전체 병합 영상",
            visible=False,
            height=400,
        )

        # ── Event wiring ─────────────────────────────────────────────

        def on_input_mode_change(mode):
            is_video = mode == "비디오로 입력"
            return gr.update(visible=is_video), gr.update(visible=not is_video)

        input_mode_radio.change(
            on_input_mode_change,
            inputs=[input_mode_radio],
            outputs=[video_mode_group, combo_mode_group],
        )

        btn_webcam_mode.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[webcam_input, upload_input],
        )
        btn_upload_mode.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[webcam_input, upload_input],
        )

        if SAMPLE_VIDEOS:
            def on_sample_select(name):
                if not name or name == _SRC_DEFAULT:
                    return gr.update(visible=False)
                return gr.update(value=SAMPLE_VIDEOS[name], visible=True)

            sample_dropdown.change(
                on_sample_select,
                inputs=[sample_dropdown],
                outputs=[sample_preview],
            )
        else:
            sample_dropdown = gr.State(None)

        def on_asset_preview_change(name, asset_map):
            if not name or name == _SRC_DEFAULT:
                return gr.update(value=None, visible=False)
            return gr.update(value=asset_map.get(name), visible=True)

        identity_asset_dd.change(
            lambda name: on_asset_preview_change(name, SAMPLE_IDENTITY),
            inputs=[identity_asset_dd],
            outputs=[identity_asset_preview],
        )
        audio_asset_dd.change(
            lambda name: on_asset_preview_change(name, SAMPLE_AUDIO),
            inputs=[audio_asset_dd],
            outputs=[audio_asset_preview],
        )
        pose_asset_dd.change(
            lambda name: on_asset_preview_change(name, SAMPLE_VIDEOS),
            inputs=[pose_asset_dd],
            outputs=[pose_asset_preview],
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

        # Run inference: append new results onto results_list_state and
        # rebuild the running merge video. Never replaces earlier results.
        run_btn.click(
            fn=lambda: "⏳ 추론 중입니다. 잠시 기다려주세요...",
            outputs=[status_md],
        ).then(
            fn=run_inference_single_page,
            inputs=[
                input_mode_radio, webcam_input, upload_input, sample_dropdown,
                identity_asset_dd, identity_upload,
                audio_asset_dd, audio_upload,
                pose_asset_dd, pose_upload,
                emotion_checkboxes, sr_checkbox,
                results_list_state,
            ],
            outputs=[results_list_state, running_merge_video, status_md],
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
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
