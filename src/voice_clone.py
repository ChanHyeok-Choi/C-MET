"""
Voice cloning with emotion control using MetaVoice + EmoKnob.

Required env var (optional):
    EMOKNOB_DIR  — path to the emoknob repo root.
                   Defaults to ../../emoknob relative to this file.
"""

import os
import sys
import pickle
import shutil
import tempfile

import torch

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

EMOKNOB_DIR = os.environ.get(
    "EMOKNOB_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../emoknob")),
)
METAVOICE_SRC = os.path.join(EMOKNOB_DIR, "src", "metavoice-src-main")
EMO_DIRS_PKL  = os.path.join(EMOKNOB_DIR, "src", "all_emo_dirs.pkl")

# ---------------------------------------------------------------------------
# Emotion name mapping: C-MET display name (base, lowercase) → emoknob key
# ---------------------------------------------------------------------------

EMOTION_NAME_MAP: dict = {
    "angry":       "angry",
    "contempt":    "contempt",
    "disgusted":   "disgust",
    "fear":        "fear",
    "happy":       "happy",
    "sad":         "sad",
    "surprised":   "surprise",
    "charismatic": "charisma",
    "desirous":    "desire",
    "empathetic":  "empathetic",
    "envious":     "envy",
    "romantic":    "romance",
    "sarcastic":   "sarcasm",
}

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-loaded)
# ---------------------------------------------------------------------------

_VC_MODELS: dict = {}
_EMO_DIRS:  dict = {}
_WHISPER_MODEL = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_metavoice_on_path() -> None:
    if METAVOICE_SRC not in sys.path:
        sys.path.insert(0, METAVOICE_SRC)


def _display_to_emo_key(display_name: str) -> str:
    """Map C-MET display name (e.g. 'charismatic [Gemini]') to emoknob pkl key."""
    base = display_name.split(" [")[0].lower()
    return EMOTION_NAME_MAP.get(base, base)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_vc_available() -> bool:
    """Return True if voice cloning can potentially run (CUDA + source files present)."""
    return (
        torch.cuda.is_available()
        and os.path.isdir(METAVOICE_SRC)
        and os.path.isfile(EMO_DIRS_PKL)
    )


def load_voice_cloning_models(output_dir: str = "tmp_vc") -> dict:
    """
    Load MetaVoice first/second stage models and speaker encoder.
    Idempotent — returns cached models on subsequent calls.
    Raises RuntimeError if CUDA is unavailable or source files are missing.
    """
    global _VC_MODELS, _EMO_DIRS

    if _VC_MODELS:
        return _VC_MODELS

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Voice cloning requires a CUDA GPU. "
            "Please run on a machine with an NVIDIA GPU."
        )
    if not os.path.isdir(METAVOICE_SRC):
        raise RuntimeError(
            f"MetaVoice source not found at '{METAVOICE_SRC}'. "
            "Set the EMOKNOB_DIR environment variable to the emoknob repo root."
        )
    if not os.path.isfile(EMO_DIRS_PKL):
        raise RuntimeError(
            f"Pre-computed emotion directions not found at '{EMO_DIRS_PKL}'. "
            "Make sure the emoknob repo is complete."
        )

    _ensure_metavoice_on_path()

    from pathlib import Path
    from huggingface_hub import snapshot_download
    from fam.llm.adapters import FlattenedInterleavedEncodec2Codebook
    from fam.llm.decoders import EncodecDecoder
    from fam.llm.fast_inference_utils import build_model
    from fam.llm.fast_inference_utils import main as vc_main
    from fam.llm.inference import (
        InferenceConfig,
        Model,
        TiltedEncodec,
        TrainedBPETokeniser,
        get_cached_embedding,
        get_enhancer,
    )
    from fam.llm.utils import get_default_dtype, normalize_text

    _dtype    = get_default_dtype()
    _device   = "cuda"
    precision = {"float16": torch.float16, "bfloat16": torch.bfloat16}[_dtype]

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading / loading MetaVoice weights from Hugging Face...")
    _model_dir = snapshot_download(repo_id="metavoiceio/metavoice-1B-v0.1")

    first_stage_adapter = FlattenedInterleavedEncodec2Codebook(end_of_audio_token=1024)

    config_second = InferenceConfig(
        ckpt_path=f"{_model_dir}/second_stage.pt",
        num_samples=1,
        seed=1337,
        device=_device,
        dtype=_dtype,
        compile=False,
        init_from="resume",
        output_dir=output_dir,
    )
    data_adapter_second = TiltedEncodec(end_of_audio_token=1024)
    llm_second_stage = Model(
        config_second,
        TrainedBPETokeniser,
        EncodecDecoder,
        data_adapter_fn=data_adapter_second.decode,
    )

    enhancer = get_enhancer("df")

    model, tokenizer, smodel, model_size = build_model(
        precision=precision,
        checkpoint_path=Path(f"{_model_dir}/first_stage.pt"),
        spk_emb_ckpt_path=Path(f"{_model_dir}/speaker_encoder.pt"),
        device=_device,
        compile=False,
        compile_prefill=False,
    )

    with open(EMO_DIRS_PKL, "rb") as f:
        _EMO_DIRS = pickle.load(f)

    _VC_MODELS = {
        "model":                model,
        "tokenizer":            tokenizer,
        "smodel":               smodel,
        "model_size":           model_size,
        "llm_second_stage":     llm_second_stage,
        "first_stage_adapter":  first_stage_adapter,
        "enhancer":             enhancer,
        "device":               _device,
        "precision":            precision,
        "output_dir":           output_dir,
        "normalize_text":       normalize_text,
        "get_cached_embedding": get_cached_embedding,
        "vc_main":              vc_main,
    }

    print(f"MetaVoice ready. Available emotion directions: {sorted(_EMO_DIRS.keys())}")
    return _VC_MODELS


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using OpenAI Whisper (base model). Returns plain text."""
    global _WHISPER_MODEL
    import whisper

    if _WHISPER_MODEL is None:
        print("Loading Whisper base model...")
        _WHISPER_MODEL = whisper.load_model("base")

    result = _WHISPER_MODEL.transcribe(audio_path)
    return result["text"].strip()


def clone_voice_with_emotion(
    text: str,
    source_audio: str,
    emotion_display_name: str,
    strength: float,
    output_path: str,
) -> str:
    """
    Generate speech that clones the source speaker's voice with a target emotion.

    Args:
        text:                 Text to synthesize (typically transcribed from source audio).
        source_audio:         Path to source .wav — provides the speaker embedding.
        emotion_display_name: C-MET display name (e.g. 'happy', 'charismatic [Gemini]').
        strength:             Emotion intensity (0.1 – 1.0).
        output_path:          Destination .wav path.

    Returns:
        output_path (same value).
    """
    if not _VC_MODELS:
        raise RuntimeError(
            "Voice cloning models not loaded. "
            "Call load_voice_cloning_models() first."
        )

    emo_key = _display_to_emo_key(emotion_display_name)
    if emo_key not in _EMO_DIRS:
        raise ValueError(
            f"Emotion '{emotion_display_name}' (resolved key='{emo_key}') not found "
            f"in pre-computed directions. Available: {sorted(_EMO_DIRS.keys())}"
        )

    m         = _VC_MODELS
    device    = m["device"]
    precision = m["precision"]

    emo_dir = torch.tensor(_EMO_DIRS[emo_key], device=device, dtype=precision)
    emo_dir = emo_dir / torch.linalg.norm(emo_dir, dim=-1, keepdim=True)

    source_emb = m["get_cached_embedding"](source_audio, m["smodel"]).to(
        device=device, dtype=precision
    )
    edited_emb = source_emb + strength * emo_dir
    text_norm  = m["normalize_text"](text)

    tokens = m["vc_main"](
        model=m["model"],
        tokenizer=m["tokenizer"],
        model_size=m["model_size"],
        prompt=text_norm,
        spk_emb=edited_emb,
        top_p=torch.tensor(0.95, device=device, dtype=precision),
        guidance_scale=torch.tensor(3.0, device=device, dtype=precision),
        temperature=torch.tensor(1.0, device=device, dtype=precision),
    )

    _text_ids, extracted_audio_ids = m["first_stage_adapter"].decode([tokens])

    wav_files = m["llm_second_stage"](
        texts=[text_norm],
        encodec_tokens=[
            torch.tensor(
                extracted_audio_ids, dtype=torch.int32, device=device
            ).unsqueeze(0)
        ],
        speaker_embs=edited_emb.unsqueeze(0),
        batch_size=1,
        guidance_scale=None,
        top_p=None,
        top_k=200,
        temperature=1.0,
        max_new_tokens=None,
    )

    raw_wav = str(wav_files[0]) + ".wav"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        m["enhancer"](raw_wav, tmp_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        shutil.copy2(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return output_path
