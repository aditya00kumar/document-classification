"""
Author: Aditya Kumar
Email: aditya00kumar@gmail.com
Description: Contains all models defined in tox.ini files to be used in the flask application.
Project: document-classification
Date Created: 18-03-2019 01:46
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "Neural Net", "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=-1),
    MLPClassifier(alpha=1),
    MultinomialNB()]


def train_model(train_vectors, train_target, model_name):
    """
    To train model based on passed data and model name
    :param train_vectors: A vector of training data after applying operation of preprocess.py
    :param train_target: Corresponding target variable for training data
    :param model_name: name of model defined in tox.ini to be trained on data
    :return: trained model
    """
    # print('model_name: ', model_name)
    for name, model in zip(names, classifiers):
        if name in model_name:
            model.fit(train_vectors, train_target)
            return model
