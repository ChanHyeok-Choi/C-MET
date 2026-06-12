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

TMP_DIR = "tmp_demo"

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

def crop_video_to_25fps(input_path: str, out_path: str):
    print("Detecting face for crop...")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=get_fa_device(),
    )

    cap = cv2.VideoCapture(input_path)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read video: {input_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    scale = 1.0
    if max(h, w) > 640:
        scale = max(h, w) / 640.0
        small = cv2.resize(frame_rgb, (int(w / scale), int(h / scale)))
    else:
        small = frame_rgb

    bboxes = fa.face_detector.detect_from_image(small[..., ::-1])
    if len(bboxes) == 0:
        raise RuntimeError(
            "No face detected in the first frame. "
            "Try trimming the video or using a different clip."
        )

    x1, y1, x2, y2 = [v * scale for v in bboxes[0][:4]]
    bw, bh = x2 - x1, y2 - y1
    pad = 0.1
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw))
    y2 = min(h, int(y2 + pad * bh))
    cw, ch = x2 - x1, y2 - y1

    cmd = (
        f'ffmpeg -i "{input_path}" '
        f'-vf "crop={cw}:{ch}:{x1}:{y1},scale=256:256" '
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


def extract_audio(input_path: str, out_path: str):
    cmd = (
        f'ffmpeg -i "{input_path}" -f wav -acodec pcm_s16le -ar 16000 '
        f'"{out_path}" -y -loglevel error'
    )
    print("Extracting audio...")
    ret = os.system(cmd)
    if ret != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            "ffmpeg audio extraction failed. "
            "Ensure the video has an audio stream and ffmpeg is installed."
        )


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

    pose_vid_target, fps = vid_preprocessing(cropped_video)
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


# ---------------------------------------------------------------------------
# Gradio-specific helpers
# ---------------------------------------------------------------------------

def get_audio_sample(display_name: str):
    """Return the path to a random .wav sample for the given emotion display name."""
    if not display_name or display_name not in EMOTION_DISPLAY_MAP:
        return None
    _, _, e2v_path = EMOTION_DISPLAY_MAP[display_name]
    audio_dir = os.path.dirname(e2v_path)
    wavs = [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.endswith(".wav")
    ]
    return random.choice(wavs) if wavs else None


def run_inference_gradio(
    video_path,
    selected_display_names,
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

    results = {}  # display_name -> save_path
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
            run_single_inference(
                identity_img, audio_wav, cropped_video, emo_e2v_path, save_path
            )
            results[display_name] = save_path
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    progress(1.0, desc="완료!")

    first_display = selected_display_names[0]
    choices = list(results.keys())
    multi = len(choices) > 1

    return (
        gr.update(value=results[first_display], visible=True),
        gr.update(choices=choices, value=first_display, visible=multi),
        results,
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
        video_state   = gr.State(None)
        results_state = gr.State({})

        with gr.Tabs() as tabs:

            # ── Stage 1: Record ───────────────────────────────────────
            with gr.Tab("① Record Video", id="tab_record"):
                gr.Markdown(
                    "웹캠으로 영상을 **촬영**하거나 파일을 **업로드**하세요.\n\n"
                    "촬영 완료 후 **Use This Video →** 버튼을 클릭하면 Stage 2로 이동합니다."
                )
                webcam_input = gr.Video(
                    sources=["webcam", "upload"],
                    label="Video Input",
                    height=400,
                )
                use_video_btn = gr.Button(
                    "Use This Video →", variant="primary", size="lg"
                )
                video_status = gr.Markdown("")

            # ── Stage 2: Select Emotions ──────────────────────────────
            with gr.Tab("② Select Emotions", id="tab_emotion"):
                gr.Markdown("감정을 **복수 선택**하고, 오디오 샘플로 미리 들어보세요.")

                emotion_checkboxes = gr.CheckboxGroup(
                    choices=EMOTION_DISPLAY_NAMES,
                    label="감정 선택 (중복 가능)",
                )
                selected_summary = gr.Markdown("선택된 감정: **없음**")

                gr.Markdown("---\n#### 오디오 미리듣기")
                with gr.Row():
                    preview_dropdown = gr.Dropdown(
                        choices=EMOTION_DISPLAY_NAMES,
                        label="미리들을 감정",
                        scale=3,
                    )
                    preview_btn = gr.Button("▶ Play Sample", scale=1)

                audio_preview = gr.Audio(
                    label="Emotion Sample",
                    autoplay=True,
                    visible=False,
                    interactive=False,
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
                    label="Result",
                    visible=False,
                    height=400,
                )
                back_to_emotion = gr.Button("← Select More Emotions")

        # ── Event wiring ─────────────────────────────────────────────

        def on_use_video(video_path):
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
            inputs=[webcam_input],
            outputs=[video_status, tabs, video_state],
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

        def on_preview(display_name):
            path = get_audio_sample(display_name)
            if path is None:
                return gr.update(visible=False)
            return gr.update(value=path, visible=True)

        preview_btn.click(
            on_preview,
            inputs=[preview_dropdown],
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
            ),
            outputs=[tabs, status_md, result_video, result_selector],
        ).then(
            fn=run_inference_gradio,
            inputs=[video_state, emotion_checkboxes],
            outputs=[result_video, result_selector, results_state],
        ).then(
            fn=lambda: "✅ 추론 완료! 아래에서 결과를 확인하세요.",
            outputs=[status_md],
        )

        # Switch result video when dropdown changes
        result_selector.change(
            fn=lambda name, res: gr.update(value=res.get(name)),
            inputs=[result_selector, results_state],
            outputs=[result_video],
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
