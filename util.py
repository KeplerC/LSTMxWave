
import librosa
import numpy as np

def load_audio(path, sample_length=64000, sample_rate=16000):
      audio, _ = librosa.load(path, sample_rate=sample_rate)
      audio = audio[:sample_length]
      return audio

# code adapted from magneta
# trim the wav based on sample length and hop length
def trim_for_encoding(wav_data, sample_length, hop_length=512):
      if wav_data.ndim == 1:
            # Max sample length is the data length
            if sample_length > wav_data.size:
                  sample_length = wav_data.size
            # Multiple of hop_length
            sample_length = (sample_length // hop_length) * hop_length
            # Trim
            wav_data = wav_data[:sample_length]
            # Assume all examples are the same length
      elif wav_data.ndim == 2:
            # Max sample length is the data length
            if sample_length > wav_data[0].size:
            sample_length = wav_data[0].size
            # Multiple of hop_length
            sample_length = (sample_length // hop_length) * hop_length
            # Trim
            wav_data = wav_data[:, :sample_length]
      return wav_data, sample_length
