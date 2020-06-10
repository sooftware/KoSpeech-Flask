import torch
from parse import parse_audio

spectrogram = parse_audio('./audio/KaiSpeech_000001.pcm')
model = torch.load('./weight_file/model.pt', map_location='cpu').module

model.listener.device = 'cpu'
model.speller.device = 'cpu'

output = model(spectrogram.unsqueeze(0), torch.IntTensor([len(spectrogram)]), teacher_forcing_ratio=0.0)
print(output)
