
import librosa
import numpy as np

def load_audio(path, sample_length=64000, sample_rate=16000):
      audio, _ = librosa.load(path, sample_rate=sample_rate)
      audio = audio[:sample_length]
      return audio

