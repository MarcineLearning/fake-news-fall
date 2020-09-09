import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding 
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import AveragePooling1D
from keras.preprocessing import sequence
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay

from embeddings import glove

root_folder = os.path.realpath('.')

def get_model_name(model_code, dataset, k):
    save_dir = 'classifiers/saved/'
    return save_dir+str(model_code)+'_model_DS'+str(dataset)+'_'+str(k)+'.h5'

"""
Bidirectional Long Short Term Memory - (RNN) Model Initialization 
"""
def build_LSTM_model(embedding_type, word2vec_model, tokenized_words, 
        embedding_dim, max_length):
    vocab_size = len(tokenized_words.items())+1
    model = Sequential()
    #embedding_matrix = compute_embedding_matrix(embedding_type, word2vec_model, 
    #embedding_dim, tokenized_words, vocab_size)
    #model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], 
    #input_length=max_length, trainable=False))
    #model.add(Input(batch_shape=(max_length, 1)))
    model.add(Bidirectional(LSTM(32, return_sequences=True,
    dropout=0.2, recurrent_dropout=0.2), input_shape=(max_length, 495)))
    model.add(TimeDistributed(Dense(8, activation='sigmoid')))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

"""
Deep Averaging Network Model Initialization
"""
def build_DAN_model(embedding_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(embedding_dim,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    return model

"""
Convolutional Neural Network Model Initialization
"""
def build_CNN_model(embedding_type, word2vec_model, 
        tokenized_words, embedding_dim, max_length):
    vocab_size = len(tokenized_words.items())+1
    model = Sequential()
    if(embedding_type!="tfidf-transformer"):    
        embedding_matrix = compute_embedding_matrix(embedding_type, 
        word2vec_model, embedding_dim, tokenized_words, vocab_size)
        model.add(Embedding(vocab_size, embedding_dim, 
        weights=[embedding_matrix], input_length=max_length, trainable=False))
    else:
        model.add(Input(batch_shape=(None, max_length, 1)))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=64, kernel_size=8 
    ,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)
    ,activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=8))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

"""
Compute Embedding Matrix for Embedding Layers
"""
def compute_embedding_matrix(embedding_type, word2vec_m, embedding_dim, 
    tokenized_words, vocab_size):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    if(embedding_type == "w2v"):
        for word, i in tokenized_words.items():
            if word in word2vec_m.wv.vocab:
                embedding_matrix[i] = word2vec_m.wv.word_vec(word)
            else:
                embedding_matrix[i] = np.random.rand(embedding_dim)
    if(embedding_type == "glove"):
        embeddings_index = glove.get_glove_vectors()
        for word, i in tokenized_words.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

"""
Build NN Model calling the corrisponding initialization function
"""
def build_neural_network(embedding_type, network_arch, w2v_model, 
        tokenized_words, embedding_dim, max_length):
    if(network_arch == 'LSTM'):
        return build_LSTM_model(embedding_type, w2v_model, tokenized_words, 
            embedding_dim, max_length)
    if(network_arch == 'DAN'):
        return build_DAN_model(embedding_dim)
    if(network_arch == 'CNN'):
        return build_CNN_model(embedding_type, w2v_model, tokenized_words, 
            embedding_dim, max_length)

"""
Return Optimizer with initialized learning schedule
"""
def get_optimizer():
    #opt = SGD(lr=0.01, momentum=0.9, decay=0.01)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return opt
