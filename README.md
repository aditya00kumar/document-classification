# document-classification
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![HitCount](http://hits.dwyl.io/aditya00kumar/document-classification.svg)](http://hits.dwyl.io/aditya00kumar/document-classification)

This project is an attempt to provide a generic pipeline for document classification using different machine learning
 models. Features of this project are following:
 - Online training of models using custom training data provided by user.
 - Define the split ratio for training and validation of result. By default 80-20 split has been provided.
 - Visualize the results and get pickle file for trained model.
 - Use pickle file of model to do prediction.

## Steps Involved
- Create flask module for including all the modules of project

1. User input in csv file having two columns namely document and category
    - [BBC Dataset](http://mlg.ucd.ie/datasets/bbc.html)
        - There are five different categories of documents namely business, entertainment, politics, sport and tech.
2. Feature Engineering
	- Removing stop words
	- Stemming and lemmatization
	- TF-IDF
	- Word2Vec
3. Model Building using default params
	- Naive-Bayes
	- SVM
	- Random forest
	- Nearest Neighbors
    - Linear SVM
    - RBF SVM
    - Gaussian Process
    - Decision Tree
    - Random Forest
    - Neural Net
    - AdaBoost
    - Naive Bayes
    - QDA
4. Results of each model.
5. Deployment of each model as an API.

##  Task List
  - Save the pickle files.
  - Make an api for pickle files to be consumed.

### Contact:
For any suggestion/clarification please contact at aditya00kumar@gmail.com

