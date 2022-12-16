import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, after_this_request
from werkzeug.utils import secure_filename
import classificator

UPLOAD_FOLDER = './temp'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 0.5 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/temp/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_src = UPLOAD_FOLDER + '/' + filename
            prediction = classificator.predict_image(classificator.load_image(img_src))
            # @after_this_request
            # def remove_file(response):
            #     os.remove(img_src)
            #     return response
            return render_template('index.html', prediction=prediction, img_src=img_src)

    return render_template('index.html', img_src='http://img1.joyreactor.cc/pics/post/автопром-ваз-лимузин-ватермарк-351083.jpeg')


app.run('127.0.0.1', port=8200, debug=True)


# Car classification
# http://img1.joyreactor.cc/pics/post/автопром-ваз-лимузин-ватермарк-351083.jpeg
# redirect(url_for('upload_file', filename=filename))