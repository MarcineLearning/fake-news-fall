import tensorflow as tf
from string import punctuation
from os import listdir
from gensim.models import Word2Vec
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import string
import collections
import re
import os
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

from classifiers import deep 
from embeddings import w2v
from utilities import metrics
from utilities import text
from utilities import report

tf.config.threading.set_inter_op_parallelism_threads(6)

EMBEDDING_DIM=300
VALIDATION_SPLIT = 0.2
MAX_LENGTH = 500

dataset=2
max_samples='NO'
max_samples=500
cv_splits = 5
model_type='LSTM'
#glove, w2v, tfidf-transformer
embedding_type="tfidf-transformer"
embedding_type='w2v'
w2v_model=0

tokenized_words, encoded_docs =[], []
news = text.get_default_dataset(dataset, max_samples)
news, documents = text.preprocess_documents(news)

if(embedding_type=='w2v' or embedding_type=='glove'):
    if(embedding_type=='w2v'):
        w2v_model = w2v.load_saved_embedding()
    tokenized_words, encoded_docs = text.tokenize_encode_documents(
    embedding_type, model_type, w2v_model, documents, 
    MAX_LENGTH, EMBEDDING_DIM)

if(embedding_type=='tfidf-transformer'):
    tokenized_words, encoded_docs = text.vectorize_transform_documents(
    news['text'])
    MAX_LENGTH=max(MAX_LENGTH, np.shape(encoded_docs)[1])
    if(model_type=="DAN"):
        EMBEDDING_DIM=max(EMBEDDING_DIM, np.shape(encoded_docs)[1])
    #EMBEDDING_DIM=1

print("encoded docs shape: ")
print(np.shape(encoded_docs))

print("="*50)
print("Model: "+model_type)
print("Embedding/Features: "+embedding_type)
print("Dataset: "+str(dataset))
print("Samples: "+str(np.shape(encoded_docs)[1]))
print("="*50)

nb_validation_samples = int(VALIDATION_SPLIT * np.shape(encoded_docs)[0])

x_train = encoded_docs[:-nb_validation_samples]
y_train = news['target'][:-nb_validation_samples]
x_test = encoded_docs[-nb_validation_samples:]
y_test = news['target'][-nb_validation_samples:]

TEST_METRICS = []
TEST_LOSS = []
best_run_history, best_precision=0,0
fold = 1
skf = StratifiedKFold(n_splits = cv_splits, random_state = 7, shuffle = True) 
opt = deep.get_optimizer()


for train_index, val_index in skf.split(x_train, y_train):
    training_data = x_train[train_index]
    validation_data = x_train[val_index]
    training_labels = y_train.iloc[train_index]
    validation_labels = y_train.iloc[val_index]

    #print('Shapes: ')
    #print(len(training_data))
    #print(len(validation_data))
    #print(len(training_labels))
    #print(len(validation_labels))

    model = deep.build_neural_network(embedding_type, model_type, w2v_model, 
    tokenized_words, EMBEDDING_DIM, MAX_LENGTH)

    model.compile(loss='binary_crossentropy', optimizer=opt, 
        metrics=[metrics.prec, 
            metrics.rec, 
            metrics.f1, 'accuracy']
    )
    print(model.summary())

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                            deep.get_model_name(model_type, dataset, fold), 
                            monitor='val_f1', 
                            verbose=0, 
							save_best_only=True, mode='max')
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', 
                            mode='min', verbose=1, patience=5)
    
    callbacks_list = [checkpoint, early_stopping]
    history = model.fit(training_data, training_labels,
			    epochs=100,
                batch_size=128,
			    callbacks=callbacks_list,
			    validation_data=(validation_data, validation_labels)
    )
	
    model.load_weights(deep.get_model_name(model_type, dataset, fold))
    results = model.evaluate(x_test, y_test)
    results = dict(zip(model.metrics_names,results))

    y_pred = model.predict_classes(x_test).ravel()
    print("="*50)
    print("==== Classification Report ====")
    print(classification_report(y_test, y_pred))
    print("="*50)    

    if(results['prec'] > best_precision):
        best_run_history = history
        best_precision = results['prec']
        print("best run precision: "+str(best_precision))

    print("Results: ")    
    print(results)
	
    TEST_METRICS.append(results)
    TEST_LOSS.append(results['loss'])

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
	
    K.clear_session()
    fold += 1

#media delle statistiche
final_stats=np.zeros((1, 4))
for vm in TEST_METRICS:
    final_stats = np.vstack([final_stats, (vm['prec'], vm['rec'], vm['f1'], vm['accuracy'])])
print("Final stats: ")
print(final_stats)
print("Average Stats Report: ")
print(np.mean(final_stats[1:], axis=0))

report.print_stats_trend(best_run_history, model_type, dataset, embedding_type)
