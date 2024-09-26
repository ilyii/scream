import torch
import torchaudio
import whisperx

device = "cuda"
compute_type = "int8"
whisperx_model = None


def load_model(cache_dir):
    global whisperx_model
    whisperx_model = whisperx.load_model(
        "large-v3", device, compute_type=compute_type, language="de", download_root=cache_dir
    )


def transcribe_voice(voice_path):
    global whisperx_model
    if whisperx_model is None:
        raise ValueError("Model not loaded.")
    text = whisperx_model.transcribe(voice_path)
    text = " ".join([x["text"] for x in text["segments"]]).strip()
    return text
