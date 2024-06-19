# README.md

# Sentiment Analysis with Logistic Regression

This project performs sentiment analysis on a large dataset of tweets using Logistic Regression. The goal is to classify tweets as either positive or negative based on their content.

# Dataset

The dataset used for this project is the "Sentiment140" dataset imported from kaggle, which contains 1.6 million tweets. Each tweet is labeled as either positive (4) or negative (0).


# Project Steps:


# (1) Environment Setup:

Install necessary libraries.
Download and prepare the dataset.

# (2) Data Preprocessing:

Load the dataset into a Pandas DataFrame.
Clean and preprocess the text data.
Perform stemming and remove stopwords.

# (3) Feature Extraction:

Convert the cleaned text data into numerical features using TF-IDF vectorization.

# (4) Model Training and Evaluation:

Split the dataset into training and testing sets.
Train a Logistic Regression model.
Evaluate the model's performance using accuracy score.


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
