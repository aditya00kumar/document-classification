"""
Author: Aditya Kumar
Email: aditya00kumar@gmail.com
Description:
Project: document_classification
Last Modified: 1/8/18 3:10 PM
"""

from flask import Flask, jsonify
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
models = config['classifiers']['models']

app = Flask(__name__)


@app.route('/')
def hello_world():
    return jsonify(models)