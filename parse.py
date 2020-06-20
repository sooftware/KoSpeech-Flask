import torch
import librosa
import numpy as np
from core import split
from const import *

DEVICES = {
    '조명': LIGHT,
    '불': LIGHT,
    '공기청정기': AIRCONDITIONER,
    '공기 청정기': AIRCONDITIONER,
    '에어컨디셔너': AIRCONDITIONER,
    '에어컨디션어': AIRCONDITIONER,
    '에어 컨디셔너': AIRCONDITIONER,
    '에어 컨디션어': AIRCONDITIONER,
    '먼지': AIRCONDITIONER,
    'None': 9
}
OBJECTS = {
    '거실': LIVING_ROOM_LIGHT,
    '안방': INNER_ROOM_LIGHT,
    '화장실': BATH_ROOM_LIGHT,
    'None': 99
}
COMMANDS = {
    '켜': ON,
    '꺼': OFF,
    '어때': STATUS,
    '자동 모드': AUTO,
    '자동모드': AUTO,
    'None': 999
}


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


def milestone(sentence):
    device, obj, command = None, None, None

    for device in DEVICES.keys():
        if device in sentence:
            device = str(DEVICES[device])
            break

    if device is None:
        device = str(DEVICES['None'])

    for obj_ in OBJECTS.keys():
        if obj_ in sentence:
            obj = str(OBJECTS[obj_])
            break

    if obj is None:
        obj = str(DEVICES['None'])

    for command_ in COMMANDS.keys():
        if command_ in sentence:
            command = str(COMMANDS[command_])
            break

    if command is None:
        command = str(COMMANDS['None'])

    order = '%s%s%s' % (device, obj, command)

    if '9' not in order:
        return order

    else:
        return None
