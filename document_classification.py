# -*- coding: utf-8 -*-
"""
Author: Aditya Kumar
Email: aditya00kumar@gmail.com
Description:
Project: Document_Classification
Last Modified: 12/21/17 11:15 AM
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from preprocess import PreProcess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


def get_data(path, generate_graph=False, display_graph=False):
    os.chdir(path)
    folders = ['business', 'entertainment', 'politics', 'sport', 'tech']
    document_category = {}
    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            with open(path + '/' + folder + '/' + file, 'rb') as f:
                data = f.read()
                # print(data.decode('cp1250'))
            document_category[data.decode('cp1250')] = folder

    df = pd.DataFrame(list(document_category.items()), columns=['Document', 'Category'])
    # print(df.head())
    plt.figure()
    df['Category'].value_counts().plot(kind='bar')
    plt.title('Distribution of Documents in Categories')
    if generate_graph:
        plt.savefig('../static/image.png', bbox_inches='tight')
    if display_graph:
        plt.show()
    # df.to_csv('../bbc_dataset.csv', index=False)
    return df


def main():
    data = get_data('/Users/aditya1/Documents/Document_Classification/bbc-dataset')

    ###############################################################################
    # Data Pre-processing steps
    ###############################################################################
    column_name = data.columns[0]
    # print(column_name)
    pre_processor = PreProcess(data, column_name)
    # todo: change code to provide all functions in class definition.
    pre_processor_operations = ['clean_html']
    data = pre_processor.clean_html()
    data = pre_processor.remove_non_ascii()
    data = pre_processor.remove_spaces()
    data = pre_processor.remove_punctuation()
    data = pre_processor.stemming()
    data = pre_processor.lemmatization()
    data = pre_processor.stop_words()

    ###############################################################################
    # Feature extraction
    ###############################################################################

    train_x, test_x, train_y, test_y = train_test_split(data.Document, data.Category, test_size=0.20)
    # print(train_x.shape, train_y.shape)
    # print(test_x.shape, test_y.shape)
    tfidf_transformer = TfidfVectorizer(min_df=1)
    train_vectors = tfidf_transformer.fit_transform(train_x)
    joblib.dump(tfidf_transformer, 'vectorizer.pkl')
    test_vectors = tfidf_transformer.transform(test_x)
    print(data.head())

    ###############################################################################
    # Perform classification with SVM, kernel=linear
    model1 = svm.SVC(kernel='linear')
    model1.fit(train_vectors, train_y)
    joblib.dump(model1, 'SVM.pkl')
    y_pred_class = model1.predict(test_vectors)
    print(metrics.accuracy_score(test_y, y_pred_class))
    print("Prediction score for classifier %s:\n%s\n" % (model1, metrics.accuracy_score(test_y, y_pred_class)))
    print("Classification report for classifier %s:\n%s\n" % (model1, metrics.classification_report(test_y,
                                                                                                    y_pred_class)))

    model2 = MultinomialNB()
    model2.fit(train_vectors, train_y)
    joblib.dump(model2, 'MultinomialNB.pkl')
    y_pred_class = model2.predict(test_vectors)
    print("Accuracy score:", metrics.accuracy_score(test_y, y_pred_class))
    print("Confusion Matrix for classifier %s:\n%s\n" % (model2, metrics.confusion_matrix(test_y, y_pred_class)))
    print("Classification report for classifier %s:\n%s\n" % (model2, metrics.classification_report(test_y, y_pred_class)))

    # pipeline = FeatureUnion([('clean', pre_processor.clean_html()), ('spaces', pre_processor.remove_spaces())])

    # print(pipeline.fit_transform())

    # text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])


if __name__ == '__main__':
    main()

