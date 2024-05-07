

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