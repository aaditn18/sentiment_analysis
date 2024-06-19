
# Sentiment Analysis with Logistic Regression

This project performs sentiment analysis on a large dataset of tweets using Logistic Regression. The goal is to classify tweets as either positive or negative based on their content.

# Dataset
The dataset used for this project is the "Sentiment140" dataset imported from kaggle, which contains 1.6 million tweets. Each tweet is labeled as either positive (4) or negative (0).

# Project Steps

# Environment Setup:
Install necessary libraries.
Download and prepare the dataset.
# Data Preprocessing:
Load the dataset into a Pandas DataFrame.
Clean and preprocess the text data.
Perform stemming and remove stopwords.
# Feature Extraction:
Convert the cleaned text data into numerical features using TF-IDF vectorization.
# Model Training and Evaluation:
Split the dataset into training and testing sets.
Train a Logistic Regression model.
Evaluate the model's performance using accuracy score.

# Code:

# Environment Setup

!pip install kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d kazanova/sentiment140

from zipfile import ZipFile
dataset = '/content/sentiment140.zip'
with ZipFile(dataset, 'r') as raw_file:
  raw_file.extractall()
  print('done')
  
# Data Preprocessing

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# Load dataset
data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')
data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', names=['target', 'id', 'date', 'flag', 'user', 'text'], encoding='ISO-8859-1')

# Data cleaning and preprocessing
data.replace({'target': {4: 1}}, inplace=True)
port_stem = PorterStemmer()

def stem_single_review(textual_content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', textual_content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    return ' '.join(stemmed_content)

data['stemmed_content'] = data['text'].apply(stem_single_review)

# Feature Extraction:

x = data['stemmed_content'].values
y = data['target'].values

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x)

# Model Training and Evaluation:

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

# Train the model
model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

# Evaluate the model
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy score on training data: ', training_data_accuracy)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy score on test data: ', test_data_accuracy)

# Results
The model achieves an accuracy score of 85% without any overfitting.

# Conclusion
This project demonstrates a complete workflow for sentiment analysis using Logistic Regression, from data preprocessing and feature extraction to model training and evaluation. With further tuning and experimentation, the model's performance can be improved.

# Requirements
Python 3.x
pandas
numpy
scikit-learn
nltk

# Acknowledgements
The Sentiment140 dataset was used in this project, available on Kaggle.
This project uses various libraries in Python for data manipulation, natural language processing, and machine learning.
