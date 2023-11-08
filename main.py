# coding: utf_8

import os
from flask import Flask, request, redirect, render_template, flash, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np


classes = ['healthy', 'septoria leaf spot', 'leaf mold', 'bacterial spot', 'target spot', 'late blight', 'spider mites two-spotted spider mite', 'tomato mosaic virus', 'tomato yellow leaf curl virus', 'early blight']

UPLOAD_FOLDER = "uploads"
# UPLOAD_FOLDER = os.path.join(UPLOAD_IMG)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'JPG', 'PNG'])

app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model.h5', compile=False)#学習済みモデルをロード
resized_size = 64

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, target_size=(resized_size, resized_size))
            img = image.img_to_array(img)
            data = np.array([img])
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)
            predicted = result[0].argmax()
            pred_answer = "これは " + classes[predicted] + " です。"
            result = [i for i in result]

            return render_template("index.html", 
                                   answer=pred_answer,
                                   img_path="ファイル名: " + filename,
                                   pred_result=zip(classes, np.round(result[0]*100, 2)),
                                   )

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)