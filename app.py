import os
import os.path
import torch
import librosa
import soundfile as sf
from modules.const import *
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from kospeech.utils import label_to_string, id2char, EOS_token
from modules.parser import parse_audio, milestone
from modules.converter import Pcm2Wav, Wav2Pcm
from socket import socket, AF_INET, SOCK_STREAM

# Basic setting
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = './audio_to_play/'
AUDIO_TO_PLAY_PATH = os.path.join(app.config['AUDIO_FOLDER'],'uploaded_audio.wav')
show_graph = False

# Load weight file
model = torch.load('./weight_file/model.pt', map_location=DEVICE).module
model.listener.device = DEVICE
model.speller.device = DEVICE
model.eval()

# Prepare socket
light_addr = (LIGHT_RASPBERRYPI_IP, PORT)
fine_dust_addr = (FINE_DUST_RASPBERRYPI_IP, PORT)

light_socket = socket(AF_INET, SOCK_STREAM)
air_socket = socket(AF_INET, SOCK_STREAM)


# Create object
# PCM => WAV, WAV => PCM
pcm2wav = Pcm2Wav()
wav2pcm = Wav2Pcm()


def convert2pcm(wave_path, pcm_path):
    """ Convert recorded files to pcm """
    y, sr = librosa.load(wave_path, sr=32000)
    y = librosa.to_mono(y)
    y = librosa.resample(y, 32000, 16000)

    sf.write(wave_path, y, 16000, format='wav', endian='little', subtype='PCM_16')
    wav2pcm(wave_path, pcm_path)


def allowed_file(filename):
    """ Check file format """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS, filename.split('.')[1]


@app.route("/", methods=['GET', 'POST'])
def index():
    global show_graph

    # If hit play button
    if request.method == 'POST':
        if os.path.isfile(AUDIO_TO_PLAY_PATH):
            os.remove(AUDIO_TO_PLAY_PATH)

        file = request.files['file']
        uploaded_file_path = UPLOAD_FOLDER + file.filename
        is_valid, extension = allowed_file(file.filename)  # check condition

        if is_valid:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Convert format
            if extension.lower() == 'pcm':
                pcm2wav(uploaded_file_path, AUDIO_TO_PLAY_PATH)
            elif extension.lower() == 'wav':
                convert2pcm(uploaded_file_path, AUDIO_TO_PLAY_PATH)

            # Extract feature & Inference by model
            spectrogram = parse_audio('./uploaded_audio/%s' % filename)
            output = model(spectrogram.unsqueeze(0), torch.IntTensor([len(spectrogram)]), teacher_forcing_ratio=0.0)[0]
            logit = torch.stack(output, dim=1).to(DEVICE)
            y_hat = logit.max(-1)[1]
            prediction = label_to_string(y_hat, id2char, EOS_token)
            os.remove(uploaded_file_path)

            # Determine destination device & command
            order = milestone(prediction[0])
            print(order)

            if order is not None:
                # Launch socket to light
                if order[0] == str(LIGHT):
                    try:  # If not connected
                        light_socket.connect(light_addr)
                        light_socket.send(order[1:].encode())
                    except:  # If already connected
                        light_socket.send(order[1:].encode())

                # Launch socket to airconditioner
                elif order[0] == str(AIRCONDITIONER):
                    try:  # If not connected
                        air_socket.connect(fine_dust_addr)
                        air_socket.send(order[3:].encode())
                    except:  # If already connected
                        air_socket.send(order[3:].encode())

                    # Receive ack (acknowledge)
                    ack = air_socket.recv(65535).decode()
                    print(ack)

                # Show graph
                elif order[0] == str(GRAPH):
                    if order[3:] == str(SHOW):
                        show_graph = True
                    elif order[3:] == str(OFF):
                        show_graph = False

            # Play page
            if show_graph:
                return render_template('display_graph.html',
                                       audio_path='.%s' % AUDIO_TO_PLAY_PATH,
                                       prediction=str(prediction[0]))
            else:
                return render_template('uploaded.html',
                                       audio_path='.%s' % AUDIO_TO_PLAY_PATH,
                                       prediction=str(prediction[0]))
    # Root page
    return render_template('homepage.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
