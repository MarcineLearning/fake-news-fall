import string
import collections
import re
import os
import copy
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn import feature_extraction
import spacy

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from embeddings import glove


STOPWORDS = set(stopwords.words('english'))
nlp = spacy.load('en', disable=['ner', 'parser'])
root_folder = os.path.realpath('.')

"""
Performs basic text cleaning: makes text lowercase,
removes symbols, words containing numbers, punctuation, text in square brackets
"""
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    # Remove a sentence if it is only one word long
    if len(text) > 2:
        return ' '.join(word for word in text.split() if word not in STOPWORDS)

"""
Implements lemmatization on articles words
"""    
def lemmatizer(text):        
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)

"""
Applies text-cleaning and lemmatization, removes empty samples, shuffles samples.
"""
def preprocess_documents(news):
    print("Preprocessing "+"="*(50-len("Preprocessing ")))
    print("Initial Dataframe Size: ")
    print(np.shape(news))
    news = news.replace(to_replace='None', value=np.nan).dropna()
    news = news.replace(to_replace='', value=np.nan).dropna()
    news = news.sample(frac=1)

    news['text'] = pd.DataFrame(news.text.apply(lambda x: clean_text(x)))
    news = news.replace(to_replace='None', value=np.nan).dropna()
    news = news[news['text'].map(lambda x: len(x.strip())>50)]
    news['text'] = news.apply(lambda x: lemmatizer(x['text']), axis=1)
    news['text'] = news['text'].str.replace('-PRON-', '')

    print("After Preprocessing Dataframe Size: ")
    print(np.shape(news))
    print("Class Population: ")
    count_fake = news[news["target"]==1].count()["text"]
    count_true = news[news["target"]==0].count()["text"]
    print("Real (True) News Samples: "+str(count_true)+" / "+str(len(news)))
    print("Fake News Samples: "+str(count_fake)+" / "+str(len(news)))
    cleaned_documents = [row.split() for row in news['text']]
    print("Documents  Shape")
    print(len(cleaned_documents))
    return news, cleaned_documents

"""
Tokenizes and encode Documents to integer arrays
"""
def tokenize_encode_documents(embedding_type, network_arch, word2vec_m, 
    documents, max_length, embedding_dim):
    if(network_arch in ['LSTM', 'CNN']):
        return tokenize_encode_pad_documents(documents, max_length)
    if(network_arch in ["DAN", "KNN", "SVM", "LINEAR-SVM", "LOG-REG"]):
       return [], compute_averaged_doc_vectors(embedding_type, documents, 
        word2vec_m, embedding_dim)

"""
Tokenizes, encodes, pads documents vectors for CNN, LSTM models
"""
def tokenize_encode_pad_documents(documents, max_length):
    t = Tokenizer()
    t.fit_on_texts(documents)
    encoded_docs = t.texts_to_sequences(documents)
    print("Encoding "+"="*50)
    print("Encoded Documents  Shape")
    print(np.shape(encoded_docs))
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print("Padded Encoded Documents  Shape")
    print(np.shape(padded_docs))
    return t.word_index, padded_docs

"""
Computes Averaged Document Vectors for DAN model
"""
def compute_averaged_doc_vectors(embedding_type, documents, 
    word2vec_m, embedding_dim):
    if(embedding_type == "w2v"):
        return compute_w2v_vectors(documents, word2vec_m, embedding_dim)
    if(embedding_type == "glove"):
        return compute_glove_vectors(documents, embedding_dim)

"""
Computes Averaged Document Vectors using Glove Embedding
"""
def compute_glove_vectors(documents, embedding_dim):
    glove_vectors = glove.get_glove_vectors()
    docs_to_index=np.zeros(embedding_dim)
    for doc in documents:
        doc_to_index=np.zeros(embedding_dim)
        for word in doc:
            if glove_vectors.get(word) is not None:
                word_to_index=np.zeros(embedding_dim)
                word_to_index = glove_vectors.get(word)
                doc_to_index = np.vstack([doc_to_index , np.array(copy.copy(word_to_index))])
            else:
                doc_to_index = np.vstack([doc_to_index , np.random.rand(embedding_dim)])
        docs_to_index = np.vstack([docs_to_index, np.mean(doc_to_index, axis=0)])
    return docs_to_index[1:]

"""
Computes Averaged Document Vectors using Word2Vec Embedding
"""
def compute_w2v_vectors(documents, word2vec_m, embedding_dim):
    docs_to_index=np.zeros(embedding_dim)
    for doc in documents:
        doc_to_index=np.zeros(embedding_dim)
        for word in doc:
            if word in word2vec_m.wv.vocab:
                word_to_index=np.zeros(embedding_dim)
                word_to_index = word2vec_m.wv.word_vec(word)
                doc_to_index = np.vstack([doc_to_index , np.array(copy.copy(word_to_index))])
            else:
                doc_to_index = np.vstack([doc_to_index , np.random.rand(embedding_dim)])
        docs_to_index = np.vstack([docs_to_index, np.mean(doc_to_index, axis=0)])
    return docs_to_index[1:]

"""
Load default project fake news datasets from Kaggle
"""
def get_default_dataset(dataset, max_samples):
    if(dataset == 1):
        data_fake = pd.read_csv(root_folder+
            '/datasets/ISOT-dataset/Fake.csv', encoding='utf-8')
        data_true = pd.read_csv(root_folder+
            '/datasets/ISOT-dataset/True.csv', encoding='utf-8')
        data_fake['target'] = 1
        data_true['target'] = 0
        news = pd.concat([data_true, data_fake]).reset_index(drop=True)
        
    if(dataset == 2):
        real_or_fake = pd.read_csv(root_folder+
            '/datasets/secondary-dataset/real-and-fake.csv', encoding='utf-8', sep=",")
        real_or_fake['target']=0
        real_or_fake.loc[real_or_fake['label'] == 'FAKE', ['target']] = 1
        news = real_or_fake
    
    if(max_samples != 'NO'):
            news = news.sample(frac=1)
            news=news[:max_samples]
    return news

"""
Vectorize and transform news articoles using TF-IDF transformer
"""
def vectorize_transform_documents(news_articles):
    #features=20000
    ngram = (1, 3)
    max_doc_freq=0.4
    min_doc_freq=0.05
    vectorizer = feature_extraction.text.CountVectorizer(
                #max_features=features,
                lowercase=True, analyzer='word',
                #preprocessor=custom_preprocessor                 
                #stop_words= stopwords_normalized
                stop_words= 'english',
                ngram_range=ngram, max_df=max_doc_freq, min_df=min_doc_freq)
    print("Fitting CountVectorizer... (BOW)")
    """
    for i in range(0,int(len(news['text']) / 2000)):
        print(str(i*2000)+" Documents Fitted")
        vectorizer.fit(news['text'][i:min(i+2000, len(news['text']))])
    """
    vectorizer.fit(news_articles)
    vectorized = vectorizer.transform(news_articles)
    transformer = feature_extraction.text.TfidfTransformer()
    transformed = transformer.fit_transform(vectorized)

    #features = vectorizer.get_feature_names() 
    #dictionary = list(map(lambda row:dict(zip(features,row)),vectorized.toarray()))

    encoded_docs = np.matrix(transformed.toarray())
    return vectorizer.vocabulary_, encoded_docs

