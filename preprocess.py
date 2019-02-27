"""
Author: Aditya Kumar
Email: aditya00kumar@gmail.com
Description: Contains all text pre-processing functions
Project: Document_Classification
Last Modified: 1/8/18 3:10 PM
"""

import nltk
import re
import html
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


class PreProcess:
    '''This class contains all text pre-processing function
    # Input parameters: Dataframe, Column_name on which function needs to be applied
    # Output parameters: Return dataframe after applying operations
    '''
    # todo: Pass functions as a list of arguments to apply in the class
    # todo: make set of words before applying all operations to reduce processing time.
    def __init__(self, data, column_name):
        self.data = data
        self.column_name = column_name
        self.stemmer = PorterStemmer()
        self.lemmatiser = WordNetLemmatizer()
        # pass

    def remove_non_ascii(self):
        self.data[self.column_name] = self.data[self.column_name].apply(lambda x: "".join(i for i in x if ord(i) < 128))
        return self.data

    def clean_html(self):
        """remove html entities"""
        self.data[self.column_name] = self.data[self.column_name].apply(html.unescape)
        return self.data

    def remove_spaces(self):
        self.data[self.column_name] = self.data[self.column_name].apply(lambda x: x.replace('\n', ' '))
        self.data[self.column_name] = self.data[self.column_name].apply(lambda x: x.replace('\t', ' '))
        self.data[self.column_name] = self.data[self.column_name].apply(lambda x: x.replace('  ', ' '))
        self.data[self.column_name] = self.data[self.column_name].apply(lambda x: x.lower())
        return self.data

    def remove_punctuation(self):
        tr = str.maketrans("", "", string.punctuation)
        # self.data[self.column_name] = self.data[self.column_name].apply(lambda x: " ".join([item.translate(tr)
        #                                                                 for item in x.split()]))
        self.data[self.column_name] = self.data[self.column_name].apply(lambda x: x.translate(tr))
        return self.data

    def stemming(self):
        # todo: provide option of selecting stemmer.
        snowball_stemmer = SnowballStemmer('english')
        # self.data[self.column_name] = self.data[self.column_name].apply(lambda x: " ".join([snowball_stemmer.stem(item)
        #                                                                 for item in x.split()]))
        self.data[self.column_name] = self.data[self.column_name].apply(lambda x: " ".join([self.stemmer.stem(item)
                                                                        for item in x.split()]))
        return self.data

    def lemmatization(self):
        self.data[self.column_name] = self.data[self.column_name].apply(lambda x: " ".join([self.lemmatiser.lemmatize(item)
                                                                        for item in x.split()]))
        return self.data

    def stop_words(self):
        stop = stopwords.words('english')
        self.data[self.column_name] = self.data[self.column_name].apply(lambda x: " ".join(set([item for item in x.split() if
                                                                                       item not in stop])))
        return self.data
