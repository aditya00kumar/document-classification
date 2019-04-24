"""
Author: Aditya Kumar
Email: aditya00kumar@gmail.com
Description: App file for flask module contains functionality of training the classifier by just uploading the
training file in user interface and selecting the proper parameters.
Project: document_classification
Last Modified: 1/8/18 3:10 PM
"""

import ast
import configparser
import os
from pathlib import Path

import nltk
import pandas as pd
from flask import Flask, render_template, request
from flask_session import Session
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from ml_models import train_model
from preprocess import PreProcess

nltk.data.path.append('./nltk_data/')

app = Flask(__name__)
app.config.from_object('config.Config')
source_path = Path(Path(os.getcwd()))
app.config['UPLOAD_FOLDER'] = os.path.join(str(source_path), 'Uploads')
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

# Read data from configuration file
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
    return render_template('index.html', files=files, classifiers=ast.literal_eval(config['classifiers']))


def read_process_data(path, files_path):
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
    vectorizer_path = os.path.join(files_path, 'tfidf_vectorizer.pk')
    joblib.dump(tfidf_transformer, vectorizer_path)
    return train_vectors, train_y


@app.route('/train', methods=['POST'])
def train():
    clfs = request.form.getlist('classifier_checked')
    print('classifier_checked', clfs)
    session_id = request.cookies['session']
    files_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    # todo: check for csv files only
    csv_file = os.listdir(files_path)[0]
    print(files_path + '/' + csv_file)
    train_vectors, train_y = read_process_data(files_path + '/' + csv_file, files_path)
    path_to_source = Path(os.getcwd())
    model_path = os.path.join(str(Path(path_to_source)), 'Uploads')

    # todo: Add multiprocessing for multiple models

    for clf in clfs:
        model = train_model(train_vectors, train_y, clf)
        print(model)
        if model:
            joblib.dump(model, Path(Path(model_path), session_id, clf.replace(" ", "_") + '.pkl'))
            print('done saving pkl')
        else:
            print('None object returned')
    files = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], session_id))
    return render_template('index.html', files=files, classifiers=ast.literal_eval(config['classifiers']))


@app.route('/test', methods=['POST'])
def bbc_data():
    return render_template('user_input.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    session_id = request.cookies['session']
    if request.method == 'POST':
        file = request.files['csv']
        print(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], session_id, file.filename))
    files = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], session_id))
    return render_template('index.html', files=files, classifiers=ast.literal_eval(config['classifiers']))


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
    return render_template('index.html', files=files, classifiers=ast.literal_eval(config['classifiers']))


@app.route('/display_classifier', methods=['GET', 'POST'])
def display_classifier():
    return render_template('index.html', classifiers=ast.literal_eval(config['classifiers']))


@app.route('/check', methods=['GET', 'POST'])
def check():
    print(request.form['file_name'])
    session_id = request.cookies['session']
    return session_id


@app.route('/submit', methods=['POST'])
def submit():
    if request.form['text_input'] == "" or len(request.form['text_input']) < 10:
        return "Please provide input large enough, Classifier can understand :)"

    # todo: change column name to be dynamically taken from training file
    test_data = pd.DataFrame([request.form['text_input']], columns=['Document'])
    session_id = request.cookies['session']
    path = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    trained_classifier = [i for i in os.listdir(path) if '.pkl' in i]
    vectorizer = os.path.join(path, 'tfidf_vectorizer.pk')
    tfidf_transformer = joblib.load(vectorizer)
    pre_processor = PreProcess(test_data, column_name='Document')
    test_data = pre_processor.clean_html()
    test_data = pre_processor.remove_non_ascii()
    test_data = pre_processor.remove_spaces()
    test_data = pre_processor.remove_punctuation()
    test_data = pre_processor.stemming()
    test_data = pre_processor.lemmatization()
    test_data = pre_processor.stop_words()
    test_data1 = tfidf_transformer.transform(test_data.Document)
    result = {}
    for clf in trained_classifier:
        model = joblib.load(os.path.join(path, clf))
        print(clf, model.predict(test_data1)[0])
        classifier_name = clf.split('/')[-1].split('.')[0]
        result[classifier_name] = model.predict(test_data1)[0]
        print(result)
    return render_template('results.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False, threaded=True)
    # app.run(host='localhost', debug=True)
