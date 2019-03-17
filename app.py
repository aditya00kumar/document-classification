"""
Author: Aditya Kumar
Email: aditya00kumar@gmail.com
Description:
Project: document_classification
Last Modified: 1/8/18 3:10 PM
"""

import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.externals import joblib
from flask_session import Session
from preprocess import PreProcess
import configparser
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from ml_models import train_model


app = Flask(__name__)
app.config.from_object('config.Config')
source_path = Path(Path(os.getcwd()))
app.config['UPLOAD_FOLDER'] = os.path.join(str(source_path), 'Uploads')
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)


conf = configparser.ConfigParser()
source_path = Path(Path(os.getcwd()))
conf.read(os.path.join(str(source_path), 'tox.ini'))
print(app.config)
print(conf.keys())
config = dict()
config['classifiers'] = conf['classifiers']['classifiers']


# @app.route('/')
@app.route('/index', methods=['POST', 'GET'])
def display_index():
    user_id = request.cookies
    session_id = request.cookies['session']
    files = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], session_id))
    print('user:', user_id)
    return render_template('index.html', files=files, classifiers=eval(config['classifiers']))


def read_process_data(path):
    data = pd.read_csv(path)
    column_name = data.columns[0]
    # print(column_name)
    pre_processor = PreProcess(data, column_name)
    # todo: change code to provide all functions in class definition.
    data = pre_processor.clean_html()
    data = pre_processor.remove_non_ascii()
    data = pre_processor.remove_spaces()
    data = pre_processor.remove_punctuation()
    data = pre_processor.stemming()
    data = pre_processor.lemmatization()
    data = pre_processor.stop_words()
    train_x, test_x, train_y, test_y = train_test_split(data.Document, data.Category, test_size=0.20)
    tfidf_transformer = TfidfVectorizer(min_df=1)
    train_vectors = tfidf_transformer.fit_transform(train_x)
    return train_vectors, train_y


@app.route('/train', methods=['POST'])
def train():
    clfs = request.form.getlist('classifier_checked')
    session_id = request.cookies['session']
    files_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    csv_file = os.listdir(files_path)[0]
    print(files_path+'/'+csv_file)
    train_vectors, train_y = read_process_data(files_path+'/'+csv_file)
    path_to_source = Path(os.getcwd())
    model_path = os.path.join(str(Path(path_to_source)), 'Uploads')

    # todo: Add multiprocessing for multiple models

    for clf in clfs:
        model = train_model(train_vectors, train_y, clf)
        if model:
            joblib.dump(model, Path(Path(model_path), session_id, clf.replace(" ", "_")+'.pkl'))
            print('done saving pkl')
        else:
            print('None object returned')
    return jsonify(request.form)


@app.route('/bbc_data', methods=['POST'])
def bbc_data():
    return render_template('user_input.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    session_id = request.cookies['session']
    files = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], session_id))
    if request.method == 'POST':
        file = request.files['csv']
        print(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], session_id, file.filename))
        return render_template('index.html', files=files)


@app.route('/', methods=['GET', 'POST'])
def dropdown():
    session_id = request.cookies['session']
    try:
        # Create target Directory
        os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], session_id))
        print("Directory  Created ")
    except FileExistsError:
        print("Directory already exists")
    files = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], session_id))
    print('files are', files)
    return render_template('index.html', files=files, classifiers=eval(config['classifiers']))


@app.route('/display_classifier', methods=['GET','POST'])
def display_classifier():
    return render_template('index.html', classifiers=eval(config['classifiers']))


@app.route('/check', methods=['GET', 'POST'])
def check():
    print(request.form['file_name'])
    session_id = request.cookies['session']
    return session_id


# @app.route('/submit', methods=['POST'])
# def submit():
#     if request.form['text_input'] == "":
#         return "Bingo I have handled this case, please provide input :)"
#     elif len(request.form['text_input']) < 20:
#         return "Please provide input large enough, Classifier can understand :)"
#     else:
#         filename = '/Users/aditya1/Documents/Document_Classification/bbc-dataset/MultinomialNB.pkl'
#         vectorizer = '/Users/aditya1/Documents/Document_Classification/bbc-dataset/vectorizer.pkl'
#         model = joblib.load(filename)
#         model2 = joblib.load('/Users/aditya1/Documents/Document_Classification/bbc-dataset/SVM.pkl')
#         data_check = pd.DataFrame([request.form['text_input']], columns=['Document'])
#         tfidf_transformer = joblib.load(vectorizer)
#         pre_processor = PreProcess(data_check, column_name='Document')
#         data_check = pre_processor.clean_html()
#         data_check = pre_processor.remove_non_ascii()
#         data_check = pre_processor.remove_spaces()
#         data_check = pre_processor.remove_punctuation()
#         data_check = pre_processor.stemming()
#         data_check = pre_processor.lemmatization()
#         data_check = pre_processor.stop_words()
#         data_check_1 = tfidf_transformer.transform(data_check.Document)
#
#         result = {model: model.predict(data_check_1)[0], model2: model2.predict(data_check_1)[0]}
#         # return 'Class Prediction is: {}'.format(model.predict(data_check_1))
#         # return result
#         return render_template('results.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # app.run(host='localhost', debug=True)
