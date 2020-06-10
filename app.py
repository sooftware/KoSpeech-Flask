import os
import torch
from flask import Flask, request
from werkzeug import secure_filename
from e2e.modules.global_ import label_to_string, id2char, EOS_token
from parse import load_audio, parse_audio, SAMPLE_RATE

UPLOAD_FOLDER = './audio/'
ALLOWED_EXTENSIONS = {'pcm'}
DEVICE = 'cpu'
IMAGE_SRC = "https://user-images.githubusercontent.com/42150335/83557467-ae625980-a54c-11ea-97d0-071f355a9743.png"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            spectrogram, sound = parse_audio('./audio/%s' % filename)
            model = torch.load('./weight_file/model.pt', map_location=DEVICE).module
            model.listener.device = DEVICE
            model.speller.device = DEVICE

            output = model(spectrogram.unsqueeze(0), torch.IntTensor([len(spectrogram)]), teacher_forcing_ratio=0.0)
            logit = torch.stack(output, dim=1).to(DEVICE)
            hypothesis = logit.max(-1)[1]
            prediction = label_to_string(hypothesis, id2char, EOS_token)
            return """
                   <!doctype html>
                   <title>Team Kai.Lib</title>
                    
                   <div style="TEXT-ALIGN: center">
                   <h1>A.I Dictation Solution</h1>
                   </div> 
                    
                   <div style="TEXT-ALIGN: center">
                   <img src=%s>
                   </div> 
                    
                   <div style="TEXT-ALIGN: center">
                   <form action="" method=post enctype=multipart/form-data>
                     <p><input type=file name=file value=Choose>
                        <input type=submit value=Predict>
                   </form>
                   </div> 
                   <p style="TEXT-ALIGN: center">
                   %s
                   </p>
                   """ % (IMAGE_SRC, str(prediction[0]))
    return """
    <!doctype html>
    <title>Team Kai.Lib</title>
    
    <div style="TEXT-ALIGN: center">
    <h1>A.I Dictation Solution</h1>
    </div> 
    
    <div style="TEXT-ALIGN: center">
    <img src=%s>
    </div> 
    
    <div style="TEXT-ALIGN: center">
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file value=Choose>
         <input type=submit value=Predict>
    </form>
    </div> 
    """ % IMAGE_SRC


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
