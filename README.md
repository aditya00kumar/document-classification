# document-classification
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
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
    - Decision Tree
    - Random Forest
    - Neural Net
4. Results of each model.
5. Deployment of each model as an API.

## Running on AWS
1. Install all the packages using `pip3 install -r requirements.txt`
2. `sudo python3 -m nltk.downloader -d /usr/share/nltk_data`
3. Then download all-corpora and stopwords
4. Run the flask app using `sudo python3 -m flask run --host=0.0.0.0 --port=80`.
5. App can be opened on public ip provided in aws console.

## Using gunicorn and nginx
6. `pip install gunicorn`
7. `sudo apt-get install nginx`
8. To start nginx server `sudo /etc/init.d/nginx start`
9. `sudo apt-get install gunicorn3`
10. `gunicorn app:app -b localhost:8000 &`


##  Task List
  - [X] Save the pickle files.
  - Make an api for pickle files to be consumed.

### Contact:
For any suggestion/clarification please contact at aditya00kumar@gmail.com

### References:
 - https://pyliaorachel.github.io/blog/tech/system/2017/07/07/flask-app-with-gunicorn-on-nginx-server-upon-aws-ec2-linux.html
 - http://nginx.org/en/docs/beginners_guide.html#conf_structure
 - https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-14-04
 
