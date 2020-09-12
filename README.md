# fake-news-fall
This repository contains the code resources as part of the project for the Machine Learning &amp; Pattern Recognition course. The code is developed in Python 3.7 and covers the implementation of different fake news classifiers.

As the repository name itself reveals, the project's focus is all about detecting fake news and confirming good news.
The training relies on two different datasets found on Kaggle.

The classifiers we implemented, trained, and tested, are the following:
  - SVM (linear/non-linear kernels)
  - Logistic Regression
  - KNN
  - Deep Learning (FeedForward, Convolutional)
  
Every classifier is applied with different features extraction methods:
  - Gensim Word2Vec embedding trained on the 2 datasets
  - GloVe embedding (pre-trained and downloaded, link below)
  - TF-IDF


# Usage:
Code developed in a Conda environment, package installed are found in the file env.txt

-> to run the training + testing demo for the SVM, Log Reg and K-NN Classifiers: run script train_test_classifiers.py

-> to run the training + testing demo for the deep learning classifiers: run script train_test_nn_classifiers.py

GloVe 6b Embeddings are not included and need to be download at (822 MB download):
http://nlp.stanford.edu/data/glove.6B.zip
