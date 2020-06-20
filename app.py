import os
import os.path
from socket import socket, AF_INET, SOCK_STREAM
import torch
import librosa
import soundfile as sf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from kospeech.utils import label_to_string, id2char, EOS_token
from parse import parse_audio, milestone
from converter import Pcm2Wav, Wav2Pcm
from const import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = './audio_to_play/'
AUDIO_TO_PLAY_PATH = os.path.join(app.config['AUDIO_FOLDER'],'uploaded_audio.wav')

model = torch.load('./weight_file/model.pt', map_location=DEVICE).module
model.listener.device = DEVICE
model.speller.device = DEVICE
model.eval()

addr = (LIGHT_RASPBERRYPI_IP, PORT)
light_socket = socket(AF_INET, SOCK_STREAM)

pcm2wav = Pcm2Wav()
wav2pcm = Wav2Pcm()


def local2pcm(wave_path, pcm_path):
    y, sr = librosa.load(wave_path, sr=32000)
    y = librosa.to_mono(y)
    y = librosa.resample(y, 32000, 16000)

    sf.write(wave_path, y, 16000, format='wav', endian='little', subtype='PCM_16')
    wav2pcm(wave_path, pcm_path)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS, filename.split('.')[1]


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if os.path.isfile(AUDIO_TO_PLAY_PATH):
            os.remove(AUDIO_TO_PLAY_PATH)

        file = request.files['file']
        uploaded_file_path = UPLOAD_FOLDER + file.filename
        is_valid, extension = allowed_file(file.filename)

        if is_valid:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            if extension.lower() == 'pcm':
                pcm2wav(uploaded_file_path, AUDIO_TO_PLAY_PATH)
            elif extension.lower() == 'wav':
                local2pcm(uploaded_file_path, AUDIO_TO_PLAY_PATH)

            spectrogram, sound = parse_audio('./uploaded_audio/%s' % filename)

            output = model(spectrogram.unsqueeze(0), torch.IntTensor([len(spectrogram)]), teacher_forcing_ratio=0.0)[0]
            logit = torch.stack(output, dim=1).to(DEVICE)
            hypothesis = logit.max(-1)[1]
            prediction = label_to_string(hypothesis, id2char, EOS_token)
            os.remove(uploaded_file_path)
            order = milestone(prediction[0])
            print(order)

            if order is not None:
                if order[0] == str(LIGHT):
                    try:
                        light_socket.connect(addr)
                        light_socket.send(order[1:].encode())
                    except:
                        light_socket.send(order[1:].encode())
                elif order[0] == AIRCONDITIONER:
                    print("AIRCONDITIONER")

            return render_template('uploaded_test.html',
                                   audio_path='.%s' % AUDIO_TO_PLAY_PATH,
                                   prediction=str(prediction[0]))
    return render_template('homepage_new.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
