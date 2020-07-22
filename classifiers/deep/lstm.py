import spacy
import tensorflow as tf
from nltk.corpus import stopwords
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding 
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import AveragePooling1D
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dropout 
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.regularizers import l2
from string import punctuation
from os import listdir
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#Pandas
import pandas as pd
#Numpy
import numpy as np
import string
import collections
#from collections import Counter
import re
import copy


import tensorflow as tf
#tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(6)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    # Remove a sentence if it is only one word long
    if len(text) > 2:
        return ' '.join(word for word in text.split() if word not in STOPWORDS)
    
def lemmatizer(text):        
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)   

data_fake = pd.read_csv('Fake_short.csv', encoding='utf-8')
data_true = pd.read_csv('True_short.csv', encoding='utf-8')
#data_fake = pd.read_csv('Fake_2.csv', encoding='utf-8')
#data_true = pd.read_csv('True_2.csv', encoding='utf-8')
#data_fake = pd.read_csv('Fake_medium.csv', encoding='utf-8')
#data_true = pd.read_csv('True_medium.csv', encoding='utf-8')

fake_test = pd.read_csv('GettingReal/top10000_getting_real_about_fake_news.csv', encoding='utf-8', sep=",")
#fake_test = pd.read_csv('GettingReal/getting_real_about_fake_news.csv', encoding='utf-8', sep=",")
#fake_true_test = pd.read_csv('RealOrFake/top10000_fake_or_real_news.csv', encoding='utf-8')
fake_true_test = pd.read_csv('RealOrFake/fake_or_real_news.csv', encoding='utf-8')
fake_test['target'] = 0
fake_true_test['target'] = 1
fake_true_test.loc[fake_true_test['label'] == 'FAKE', ['target']] = 0

data_fake['target'] = 0
data_true['target'] = 1
news = pd.concat([data_true, data_fake])
news = news.sample(frac=1).reset_index(drop=True)

news['text'] = pd.DataFrame(news.text.apply(lambda x: clean_text(x)))
nlp = spacy.load('en', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
news['text'] =  news.apply(lambda x: lemmatizer(x['text']), axis=1)
news['text'] = news['text'].str.replace('-PRON-', '')
#x_train,x_test,y_train,y_test = model_selection.train_test_split(news_clean['text'], news.target, test_size=0.3)
documents = [row.split() for row in news['text']]
print("Documents  Shape")
print(len(documents))

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(documents)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(documents)

#testing set supplementari
test_encoded_fake = t.texts_to_sequences(fake_test['text'])
test_encoded_fake_true = t.texts_to_sequences(fake_true_test['text'])
print("Encoded Documents  Shape")
print(np.shape(encoded_docs))
# pad documents to a max length of 4 words
# pad documents to a max length 
max_length = 0
for ed in encoded_docs:
    if len(ed) > max_length:
        max_length= len(ed)

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print("Padded Encoded Documents  Shape")
print(np.shape(padded_docs))

EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

nb_validation_samples = int(VALIDATION_SPLIT * np.shape(padded_docs)[0])
print("validation samples")
print(nb_validation_samples)
x_train = padded_docs[:-nb_validation_samples]
y_train = news['target'][:-nb_validation_samples]
x_test = padded_docs[-nb_validation_samples:]
y_test = news['target'][-nb_validation_samples:]


x_fake_test_pad=pad_sequences(test_encoded_fake, maxlen=max_length, padding='post')
x_fake_true_test_pad=pad_sequences(test_encoded_fake_true, maxlen=max_length, padding='post')
samples_fake = int(VALIDATION_SPLIT * np.shape(x_fake_test_pad)[0])
samples_fake_true = int(VALIDATION_SPLIT * np.shape(x_fake_true_test_pad)[0])

x_fake_val=x_fake_test_pad[-samples_fake:]
y_fake_val= fake_test['target'][-samples_fake:]
x_fake_true_val=x_fake_true_test_pad[-samples_fake_true:]
y_fake_true_val=fake_true_test['target'][-samples_fake_true:]

x_fake_test=x_fake_test_pad[:-samples_fake]
y_fake_test= fake_test['target'][:-samples_fake]
x_fake_true_test=x_fake_true_test_pad[:-samples_fake_true]
y_fake_true_test=fake_true_test['target'][:-samples_fake_true]

x_validation = np.concatenate((x_fake_val, x_fake_true_val), axis=0)
y_validation = np.concatenate((y_fake_val, y_fake_true_val), axis=0)

#MAX_VOCAB_SIZE= 1000

w2v_model = Word2Vec(documents, 
                     min_count=20,
                     window=200,
                     #window=10,
                     size=EMBEDDING_DIM,
                     workers=6)
                     
#w2v_model.build_vocab(documents)
w2v_model.train(documents, total_examples=w2v_model.corpus_count, epochs=10)
w2v_model.init_sims(replace=True)


# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in t.word_index.items():
    if word in w2v_model.wv.vocab:
        embedding_matrix[i] = w2v_model.wv.word_vec(word)
    else:
        embedding_matrix[i] = np.random.rand(EMBEDDING_DIM)

print("input shape: ")
print(np.shape(x_train))
print(np.shape(np.asarray(x_train[0])))


model = Sequential()
e = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(e)
# define LSTM
model.add(Bidirectional(LSTM(32, return_sequences=True)))
#model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# summarize the model
print(model.summary())
# fit the model
model.fit(x_train, y_train, epochs=10)   #,validation_data=(x_validation, y_validation))

# evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %f' % (accuracy*100))

print("-"*50)
# make a prediction
y_fake_new = model.predict_classes(x_fake_test)
# show the inputs and predicted outputs
print("Test Set Pure Fake News: ")
guessed,misclassified=0,0

for i in range(len(x_fake_test)):
    if(y_fake_test[i] == y_fake_new[i]):
        guessed = guessed + 1
    else:
        misclassified = misclassified + 1
print("Corretti: "+str(guessed) +" su "+str(len(x_fake_test)))
print("Misclassified: "+str(misclassified) +" su "+str(len(x_fake_test)))
accuracy = 100*guessed/len(x_fake_test)
print("Accuracy: "+str(accuracy))

print("-"*50)
# make a prediction
y_fake_true_new = model.predict_classes(x_fake_true_test)
# show the inputs and predicted outputs
print("Test Fake Or True: ")
guessed,misclassified=0,0

for i in range(len(x_fake_true_test)):
    if(y_fake_true_test[i] == y_fake_true_new[i]):
        guessed = guessed + 1
    else:
        misclassified = misclassified + 1
print("Corretti: "+str(guessed) +" su "+str(len(x_fake_true_test)))
print("Misclassified: "+str(misclassified) +" su "+str(len(x_fake_true_test)))
accuracy = 100*guessed/len(x_fake_true_test)
print("Accuracy: "+str(accuracy))

#model.save('fake_news_cnn_word2vec', overwrite=True)
