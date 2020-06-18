import torch
import librosa
import numpy as np
from core import split


SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 320
HOP_LENGTH = 160


def load_audio(audio_path):
    """
    Load audio file (PCM) to sound. if del_silence is True, Eliminate all sounds below 30dB.
    If exception occurs in numpy.memmap(), return None.
    """
    sound = np.memmap(audio_path, dtype='h', mode='r').astype('float32')

    non_silence_indices = split(sound, top_db=30)
    sound = np.concatenate([sound[start:end] for start, end in non_silence_indices])

    sound /= 32767  # normalize audio
    return sound


def parse_audio(audio_path):
    sound = load_audio(audio_path)

    spectrogram = librosa.feature.melspectrogram(sound, SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    spectrogram = instancewise_standardization(spectrogram)

    spectrogram = spectrogram[:, ::-1]
    spectrogram = torch.FloatTensor(np.ascontiguousarray(np.swapaxes(spectrogram, 0, 1)))

    return spectrogram, sound

def instancewise_standardization(spectrogram):
    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram -= mean
    spectrogram /= std
    return spectrogram

