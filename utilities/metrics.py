import keras.backend as K
import numpy as np

def mean_squared_error(y_true, y_pred):
    return K.round(K.mean(K.square(y_pred - y_true), axis=-1))

def rec(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return K.round(recall)

def prec(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return K.round(precision)

def f1(y_true, y_pred):
    precision = prec(y_true, y_pred)
    recall = rec(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
