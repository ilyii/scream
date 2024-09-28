"""
Module for transcribing voice messages using the Whisper ASR model.

This module provides functionality to load the Whisper model and transcribe
audio files to text.
"""

import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Constants
MODEL_ID = "openai/whisper-large-v3"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Global variables
processor = None
wpipeline = None
whisper_model = None


def load_model(cache_dir: str) -> None:
    """
    Load the Whisper ASR model and set up the pipeline.

    Args:
        cache_dir (str): Directory to cache the downloaded model.

    Raises:
        RuntimeError: If there's an error loading the model.
    """
    global whisper_model, processor, wpipeline

    try:
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True, cache_dir=cache_dir
        )
        whisper_model.to(DEVICE)

        processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)

        wpipeline = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE,
            chunk_length_s=30,
        )
    except Exception as e:
        raise RuntimeError(f"Error loading Whisper model: {str(e)}")


def transcribe_voice(voice_path: str) -> str:
    """
    Transcribe a voice message to text.

    Args:
        voice_path (str): Path to the voice file to transcribe.

    Returns:
        str: Transcribed text.

    Raises:
        ValueError: If the model hasn't been loaded.
        RuntimeError: If there's an error during transcription.
    """
    if wpipeline is None:
        raise ValueError("Model not loaded. Call load_model() first.")

    try:
        result = wpipeline(voice_path, generate_kwargs={"language": "de"})
        return result["text"].strip()
    except Exception as e:
        raise RuntimeError(f"Error transcribing voice: {str(e)}")


# Uncomment and modify if ant to use whisperx in the future
# import whisperx
# whisperx_model = None
# compute_type = "int8"

# def load_whisperx_model(cache_dir):
#     global whisperx_model
#     whisperx_model = whisperx.load_model(
#         "large-v3", DEVICE, compute_type=compute_type, language="de", download_root=cache_dir
#     )

# def transcribe_with_whisperx(voice_path):
#     if whisperx_model is None:
#         raise ValueError("WhisperX model not loaded.")
#     result = whisperx_model.transcribe(voice_path)
#     return " ".join([x["text"] for x in result["segments"]]).strip()
