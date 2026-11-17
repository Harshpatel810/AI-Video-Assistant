import os
import shutil

# Ensure FFmpeg binary directory is in PATH
FFMPEG_PATHS = [
    r"C:\ffmpeg\ffmpeg-2026\bin",
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
]
for p in FFMPEG_PATHS:
    if os.path.isdir(p) and p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
        break

from dotenv import load_dotenv
load_dotenv()

import whisper
from pydub import AudioSegment

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SRAVAANI_MODEL_ID = os.getenv("SRAVAANI_MODEL_ID", "ARTPARK-IISc/SraVaani-1.0")
SRAVAANI_PIECE_SECONDS = int(os.getenv("SRAVAANI_PIECE_SECONDS", "30"))

_whisper_model = None
_sravaani_model = None


def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _whisper_model


def load_sravaani_model():
    global _sravaani_model
    if _sravaani_model is None:
        import torch
        from transformers import AutoModel

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set in environment or .env file.\n"
                f"Model '{SRAVAANI_MODEL_ID}' is a gated Hugging Face repository.\n\n"
                "To get access:\n"
                f"1. Visit https://huggingface.co/{SRAVAANI_MODEL_ID} and click 'Agree and access repository'.\n"
                "2. Create an access token at https://huggingface.co/settings/tokens\n"
                "3. Add HF_TOKEN=your_token_here to your .env file."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SraVaani model ({SRAVAANI_MODEL_ID}) on {device}...")
        _sravaani_model = AutoModel.from_pretrained(
            SRAVAANI_MODEL_ID,
            trust_remote_code=True,
            token=hf_token,
        ).to(device).eval()
        print("SraVaani model loaded.")
    return _sravaani_model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_whisper_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"]


def _extract_text_from_hyps(hyps) -> str:
    if isinstance(hyps, list):
        parts = []
        for h in hyps:
            if hasattr(h, "text"):
                parts.append(h.text)
            else:
                parts.append(str(h))
        return " ".join(parts).strip()
    elif hasattr(hyps, "text"):
        return hyps.text.strip()
    return str(hyps).strip()


def transcribe_chunk_sravaani(chunk_path: str) -> str:
    """
    Transcribe a chunk using ARTPARK-IISc/SraVaani-1.0.
    Audio longer than SRAVAANI_PIECE_SECONDS is sliced into smaller pieces
    to manage memory and ensure robust decoding.
    """
    model = load_sravaani_model()
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SRAVAANI_PIECE_SECONDS * 1000

    if len(audio) <= piece_ms:
        hyps = model.transcribe(chunk_path, return_hypotheses=True)
        return _extract_text_from_hyps(hyps)

    full_text = []
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sra_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → SraVaani piece {i + 1}/{total_pieces} ...")
            hyps = model.transcribe(piece_path, return_hypotheses=True)
            text = _extract_text_from_hyps(hyps)
            if text:
                full_text.append(text)
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(full_text).strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route audio chunk to Whisper or SraVaani depending on language:
    - english  → Whisper (local)
    - hinglish → ARTPARK-IISc/SraVaani-1.0 (local Indic ASR)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sravaani(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""
    engine = "SraVaani (ARTPARK-IISc/SraVaani-1.0)" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription complete.")
    return full_transcript.strip()
