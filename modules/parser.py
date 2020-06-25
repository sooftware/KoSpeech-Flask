import sounddevice as sd
import torch
import librosa
import numpy as np
from modules.core import split
from modules.const import *

# Hoffman encoding
# DEVICES is single digit
# OBJECTS is two digit
# COMMANDS is three digit
# Ex) 거실에 조명 불 좀 꺼줘
# => LIGHT + LIVING_ROOM_LIGHT + OFF => 017100
hop_length = int(16000 * 0.001 * 10)
DEVICES = {
    '조명': LIGHT,
    '불': LIGHT,
    '공기청정기': AIRCONDITIONER,
    '공기 청정기': AIRCONDITIONER,
    '에어컨디셔너': AIRCONDITIONER,
    '에어컨디션어': AIRCONDITIONER,
    '에어 컨디셔너': AIRCONDITIONER,
    '에어 컨디션어': AIRCONDITIONER,
    '에어컨디션화': AIRCONDITIONER,
    '에어 컨디션화': AIRCONDITIONER,
    '먼지': AIRCONDITIONER,
    '그래프': GRAPH,
    'None': NONE
}
OBJECTS = {
    '거실': LIVING_ROOM_LIGHT,
    '안방': INNER_ROOM_LIGHT,
    '화장실': BATH_ROOM_LIGHT,
    'None': NONE * 11
}
COMMANDS = {
    '켜': ON,
    '꺼': OFF,
    '어때': STATUS,
    '자동 모드': AUTO,
    '자동모드': AUTO,
    '자동': AUTO,
    '보여줘': SHOW,
    '보여줄래': SHOW,
    'None': NONE * 111
}


def play_audio(audio_path):
    sound, sr = librosa.core.load(audio_path, sr=SAMPLE_RATE)  # Play Audio
    sd.play(sound, 16000)


def load_audio(audio_path):
    """
    Load audio file (PCM) to sound. if del_silence is True, Eliminate all sounds below 30dB.
    If exception occurs in numpy.memmap(), return None.
    """
    signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32')
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
    spectrogram -= spectrogram.mean()

    return torch.FloatTensor(spectrogram).transpose(0, 1)


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
        obj = str(OBJECTS['None'])

    for command_ in COMMANDS.keys():
        if command_ in sentence:
            command = str(COMMANDS[command_])
            break

    if command is None:
        command = str(COMMANDS['None'])

    order = '%s%s%s' % (device, obj, command)

    if order[0] == str(LIGHT) and order not in str(NONE):
        return order

    elif order[0] == str(AIRCONDITIONER) and order[:-3] not in str(NONE):
        return order

    elif order[0] == str(GRAPH):
        if order[3:] == str(SHOW) or order[3:] == str(OFF):
            return order

    else:
        return None
