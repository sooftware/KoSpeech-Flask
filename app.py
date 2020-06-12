import os
import torch
from flask import Flask, request
from werkzeug import secure_filename
from e2e.modules.global_ import label_to_string, id2char, EOS_token
from parse import load_audio, parse_audio, SAMPLE_RATE

UPLOAD_FOLDER = './audio/'
ALLOWED_EXTENSIONS = {'pcm'}
DEVICE = 'cpu'
BACKGROUND_SRC = "https://actionpower.kr/wp-content/uploads/2017/04/asasasa.jpg"

HOMEPAGE_HTML = """
<!doctype html>
<head>
<title>KoSpeech</title>
<style>
body {
background-image: url("https://user-images.githubusercontent.com/42150335/84467650-bdb57580-acb7-11ea-817f-4122acadd1d9.png");
background-size: cover;
background-position: centoer;
}
</style>
</head>

<body>

</body>
<span style=" font: italic bold 4em/1em Times New Roman, serif ;">
<h1>KoSpeech</h1>
</span>
<div>
<form action="" method=post enctype=multipart/form-data>
 <p><input type=file name=file value=Choose>
    <input type=submit value=Predict>
</form>
</div> 
"""

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
                <head>
                <title>KoSpeech</title>
                <style>
                body {
                background-image: url("https://user-images.githubusercontent.com/42150335/84467650-bdb57580-acb7-11ea-817f-4122acadd1d9.png");
                background-size: cover;
                background-position: centoer;
                }
                </style>
                </head>
                
                <body>
                
                </body>
                <span style=" font: italic bold 4em/1em Times New Roman, serif ;">
                <h1>KoSpeech</h1>
                </span>
                <div>
                <form action="" method=post enctype=multipart/form-data>
                 <p><input type=file name=file value=Choose>
                    <input type=submit value=Predict>
                </form>
                </div> 
                <p style>
                %s
                </p>
               """ % str(prediction[0])
    return HOMEPAGE_HTML


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
