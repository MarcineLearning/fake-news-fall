import pandas as pd
import numpy as np
import string
import collections
import re
import copy
import nltk
from nltk.corpus import stopwords
from string import punctuation
from os import listdir
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, AveragePooling1D, Flatten, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold, StratifiedKFold
import spacy
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(6)
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
# disabling Named Entity Recognition for speed
nlp = spacy.load('en', disable=['ner', 'parser'])

dataset=2
save_dir = 'saved_models/'

def get_model_name(k):
    return 'DAN_model_DS'+str(dataset)+'_'+str(k)+'.h5'

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, 
    remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    # Remove a sentence if it is only one word long
    if len(text) > 2:
        return ' '.join(word for word in text.split() if word not in STOPWORDS)
    
def lemmatizer(text):
    if(text is None):
        print(text)
        print(np.shape(text))
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)   

if(dataset == 1):
    data_fake = pd.read_csv('Fake.csv', encoding='utf-8')
    data_true = pd.read_csv('True.csv', encoding='utf-8')
    data_fake['target'] = 0
    data_true['target'] = 1
    data_fake = data_fake.replace(to_replace='None', value=np.nan).dropna()
    data_true = data_true.replace(to_replace='None', value=np.nan).dropna()
    news = pd.concat([data_true, data_fake]).reset_index(drop=True)

if(dataset == 2):
    real_or_fake = pd.read_csv('fake_or_real_news.csv', encoding='utf-8', sep=",")
    real_or_fake['target']=1
    real_or_fake.loc[real_or_fake['label'] == 'FAKE', ['target']] = 0
    real_or_fake=real_or_fake[:2000]    
    news = real_or_fake

news = news.sample(frac=1)
print("Initial Dataframe Size: ")
print(np.shape(news))
print("="*50)
news = news.replace(to_replace='None', value=np.nan).dropna()
news = news.replace(to_replace='', value=np.nan).dropna()

news['text'] = pd.DataFrame(news.text.apply(lambda x: clean_text(x)))
news = news.replace(to_replace='None', value=np.nan).dropna()
news = news[news['text'].map(lambda x: len(x.strip())>50)]
news['text'] = news.apply(lambda x: lemmatizer(x['text']), axis=1)
news['text'] = news['text'].str.replace('-PRON-', '')

print("After Preprocessing Dataframe Size: ")
print(np.shape(news))
print("="*50)
print("Class Population: ")
count_fake = news[news["target"]==0].count()["text"]
count_true = news[news["target"]==1].count()["text"]
print("Real (True) News Samples: "+str(count_true)+" / "+str(len(news)))
print("Fake News Samples: "+str(count_fake)+" / "+str(len(news)))

documents = [row.split() for row in news['text']]
w2v_model = Word2Vec.load("word2vec_training/saved_embeddings/w2v_fake_news.model")

EMBEDDING_DIM=300
VALIDATION_SPLIT = 0.2

#document vector averaging
docs_to_index=np.zeros(EMBEDDING_DIM)
for doc in documents:
    doc_to_index=np.zeros(EMBEDDING_DIM)
    for word in doc:
        if word in w2v_model.wv.vocab:
            word_to_index=np.zeros(EMBEDDING_DIM)
            word_to_index = w2v_model.wv.word_vec(word)
            doc_to_index = np.vstack([doc_to_index , np.array(copy.copy(word_to_index))])
        else:
            doc_to_index = np.vstack([doc_to_index , np.random.rand(EMBEDDING_DIM)])
    docs_to_index = np.vstack([docs_to_index, np.mean(doc_to_index, axis=0)])
    
docs_to_index = docs_to_index[1:]
print("Shape Docs input ")
print(np.shape(docs_to_index))

VALIDATION_ACCURACY = []
VALIDATION_LOSS = []
#kf = KFold(n_splits = 5)
fold = 1
skf = StratifiedKFold(n_splits = 5, random_state = 7, shuffle = True) 

test_samples = int(VALIDATION_SPLIT * np.shape(docs_to_index)[0])
print("Test samples")
print(test_samples)
x_train = docs_to_index[:-test_samples]
y_train = news['target'][:-test_samples]
x_test = docs_to_index[-test_samples:]
y_test = news['target'][-test_samples:]

for train_index, val_index in skf.split(x_train, y_train):
    print("Train Indexes: "+str(train_index[0]))
    print("Validation Indexes: "+str(val_index[0]))
    print("Train Indexes: "+str(len(train_index)))
    print("Validation Indexes: "+str(len(val_index)))
    
    training_data = x_train[train_index]
    validation_data = x_train[val_index]
    training_labels = y_train.iloc[train_index]
    validation_labels = y_train.iloc[val_index]
    
    count_fake = news[news["target"]==0].count()["text"]
    count_true = news[news["target"]==1].count()["text"]
    print("Real (True) News Samples: "+str(count_true)+" / "+str(len(news)))
    print("Fake News Samples: "+str(count_fake)+" / "+str(len(news)))

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(EMBEDDING_DIM,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', 
    metrics=['accuracy']
    #metrics=[
         #tf.keras.metrics.PrecisionAtRecall(recall=0.5),
         #tf.keras.metrics.Precision(),
         #tf.keras.metrics.Recall(),
         #tf.keras.metrics.Accuracy()]
    )
   
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold), 
							#monitor='val_precision_at_recall', 
                            monitor='val_accuracy', 
                            verbose=10, 
							save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(training_data, training_labels,
			    epochs=10,
			    callbacks=callbacks_list,
			    validation_data=(validation_data, validation_labels))
	
    #model.load_weights(save_dir+get_model_name(fold))
    results = model.evaluate(x_test, y_test)
    results = dict(zip(model.metrics_names,results))
    print("Results: ")    
    print(results)
	
    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
	
    tf.keras.backend.clear_session()
    fold += 1

"""
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(EMBEDDING_DIM,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', 
#metrics=['accuracy']
metrics=[tf.keras.metrics.PrecisionAtRecall(recall=0.5)]
)
# summarize the model
print(model.summary())
# fit the model
model.fit(x_train, y_train, epochs=10)
# evaluate the model
loss, PrecisionAtRecall = model.evaluate(x_test, y_test)
print('PrecisionAtRecall(0.5): %f' % (PrecisionAtRecall*100))
#model.save('fake_news_cnn_word2vec', overwrite=True)
"""
