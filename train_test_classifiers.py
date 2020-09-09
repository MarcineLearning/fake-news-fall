#Pandas
import pandas as pd
#Numpy
import numpy as np
#Seaborn
import seaborn as sns
#Matplotlib
import matplotlib.pyplot as plt
#SKLEARN

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from classifiers import deep 
from embeddings import w2v
from utilities import metrics
from utilities import text
from utilities import report

EMBEDDING_DIM=300
VALIDATION_SPLIT = 0.2
MAX_LENGTH = 500

dataset=2
max_samples='NO'
max_samples=500
cv_splits = 5
#KNN, SVM, LINEAR-SVM, LOG-REG, BERNOULLI
model_type='BERNOULLI'
embedding_type='glove'
w2v_model=0  

news = text.get_default_dataset(dataset, max_samples)
news, documents = text.preprocess_documents(news)

if(embedding_type=="w2v" or embedding_type=="glove"):
    if(embedding_type=="w2v"):
        w2v_model = w2v.load_saved_embedding()
    tokenized_words, encoded_docs = text.tokenize_encode_documents(
    embedding_type, model_type, w2v_model, documents, 
    MAX_LENGTH, EMBEDDING_DIM)

if(embedding_type=="tfidf-transformer"):    
    tokenized_words, encoded_docs = text.vectorize_transform_documents(news['text'])

test_samples = int(VALIDATION_SPLIT * np.shape(encoded_docs)[0])
x_train = encoded_docs[:-test_samples]
y_train = news['target'][:-test_samples]
x_test = encoded_docs[-test_samples:]
y_test = news['target'][-test_samples:]

print("="*50)
print("Model: "+model_type)
print("Embedding/Features: "+embedding_type)
print("Dataset: "+str(dataset))
print("Samples: "+str(np.shape(encoded_docs)[1]))
print("="*50)

if(model_type =="LINEAR-SVM"):
    kernel='linear'
    model = SVC(kernel=kernel, C=10.0, gamma=1.0)
if(model_type =="SVM"):
    kernel='rbf'
    model = SVC(kernel=kernel, C=10.0, gamma=1.0)
if(model_type =="KNN"):
    model = KNeighborsClassifier(n_neighbors=10)
if(model_type =="LOG-REG"):
    model = linear_model.LogisticRegression(solver = 'saga', 
    class_weight='balanced', 
    penalty='none',
    max_iter=1500)
if(model_type =="BERNOULLI"):
    model = BernoulliNB()

model.fit(x_train, y_train)

scores = model_selection.cross_val_score(model, x_train, y_train, cv=cv_splits, verbose=10)
print("Cross Validation Scores: ")
print(scores)

prediction = model.predict(x_test)

fpr, tpr, thresholds = roc_curve(y_test, prediction)
auc = auc(fpr, tpr)

print("Test Accuracy: ")
print(accuracy_score(y_test, prediction)*100)
print("="*50)
print("==== Classification Report ====")
print(classification_report(y_test, prediction))
print("="*50)   
print("Confusion Matrix: ")
print(confusion_matrix(y_test, prediction))
print("="*50)
print("="*50)   

