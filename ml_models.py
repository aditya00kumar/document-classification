"""
Author: Aditya Kumar
Email: aditya00kumar@gmail.com
Description: Contains all models defined in tox.ini files to be used in the flask application.
Project: document-classification
Date Created: 18-03-2019 01:46
"""

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


def train_model(train_vectors, train_target, model_name):
    """
    To train model based on passed data and model name
    :param train_vectors: A vector of training data after applying operation of preprocess.py
    :param train_target: Corresponding target variable for training data
    :param model_name: name of model defined in tox.ini to be trained on data
    :return: trained model
    """
    print('model_name: ', model_name)
    model = None
    for name, model in zip(names, classifiers):
        # print(name, model)
        if name in model_name:
            model.fit(train_vectors, train_target)
    return model
