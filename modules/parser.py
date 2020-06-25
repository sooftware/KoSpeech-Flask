import sounddevice as sd
import torch
import librosa
import numpy as np
from modules.core import split
from modules.const import *

hop_length = int(16000 * 0.001 * 10)


def play_audio(audio_path):
    sound, sr = librosa.core.load(audio_path, sr=SAMPLE_RATE)  # Play Audio
    sd.play(sound, 16000)


def load_audio(audio_path):
    """
    Load audio file (PCM) to sound. if del_silence is True, Eliminate all sounds below 30dB.
    If exception occurs in numpy.memmap(), return None.
    """
    signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32')

    if audio_path.lower().endswith('.wav'):
        play_audio(audio_path)

    non_silence_indices = split(signal, top_db=30)
    signal = np.concatenate([signal[start:end] for start, end in non_silence_indices])

    return signal / 32767  # normalize audio


def parse_audio(audio_path):
    signal = load_audio(audio_path)

    spectrogram = torch.stft(
        torch.FloatTensor(signal),
        N_FFT,
        hop_length=hop_length,
        win_length=N_FFT,
        window=torch.hamming_window(N_FFT),
        center=False,
        normalized=False,
        onesided=True
    )
    spectrogram = (spectrogram[:, :, 0].pow(2) + spectrogram[:, :, 1].pow(2)).pow(0.5)
    spectrogram = np.log1p(spectrogram.numpy())
    spectrogram = torch.FloatTensor(spectrogram).transpose(0, 1)
    spectrogram -= spectrogram.mean()

    return spectrogram
