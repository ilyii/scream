




import math
from matplotlib import pyplot as plt
import numpy as np
import torch
from pytube import YouTube


def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()

  
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  duration = len(waveform) / sample_rate
  time = np.linspace(0, duration, len(waveform))

  plt.figure(figsize=(20, 4))
  plt.title(title)
  plt.plot(time, waveform, linewidth=1, color='black')
  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')
  if xlim:
    plt.xlim(xlim)
  if ylim:
    plt.ylim(ylim)
  plt.show(block=False)


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  duration = len(waveform) / sample_rate
  time = np.linspace(0, duration, len(waveform))

  plt.figure(figsize=(20, 4))
  plt.title(title)
  plt.specgram(waveform, Fs=sample_rate)
  plt.xlabel('Time (s)')
  plt.ylabel('Frequency')
  if xlim:
    plt.xlim(xlim)
  plt.show(block=False)




def get_sine_sweep(sample_rate, offset=0):
  """
  Generate a sine sweep signal. The maximum sweep rate is defined as the maximum frequency change per second.
  """
  max_sweep_rate = sample_rate
  start, stop = math.log(offset), math.log(offset + max_sweep_rate // 2)
  freq =  torch.exp(torch.linspace(start, stop, sample_rate, dtype=torch.double)) - offset
  delta = 2 * math.pi * freq / sample_rate
  cummulative = torch.cumsum(delta, dim=0)
  signal = torch.sin(cummulative).unsqueeze(dim=0)
  return signal


def download_audio(url, output_path):
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_stream.download(output_path)
        print("Audio downloaded successfully.")
    except Exception as e:
        print(f"Error: {str(e)}")


def log_mel_spectrogram(audio, n_mels, padding: int = 0, device = None):

    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
