import pandas as pd
import numpy as np
import string
import re
import os
import nltk
from nltk.corpus import stopwords
from string import punctuation
from gensim.models import Word2Vec
import spacy

root_folder = os.path.realpath('.')

EMBEDDING_DIM = 300
TEXT_WINDOW=100
MIN_WORD_COUNT=50

def word2vec_training():
    data_fake = pd.read_csv('datasets/1/Fake.csv', encoding='utf-8')
    data_true = pd.read_csv('datasets/1/True.csv', encoding='utf-8')
    data_fake['target'] = 0
    data_true['target'] = 1
    data_fake = data_fake.replace(to_replace='None', value=np.nan).dropna()
    data_true = data_true.replace(to_replace='None', value=np.nan).dropna()
    dataset1 = pd.concat([data_true, data_fake]).reset_index(drop=True)

    dataset2 = pd.read_csv('datasets/2/fake_or_real_news.csv', encoding='utf-8', sep=",")
    dataset2['target']=1
    dataset2.loc[dataset2['label'] == 'FAKE', ['target']] = 0

    datasets =[dataset1, dataset2]
    ds_count = 0

    for dataset in datasets:
        print("="*50)
        print("Dataset numero "+str(ds_count))
        news = dataset
        print("** Initial ** Dataframe Size: ")
        print(np.shape(news))
        print("="*50)
        news = news.replace(to_replace='None', value=np.nan).dropna()
        news = news.replace(to_replace='', value=np.nan).dropna()

        news['text'] = pd.DataFrame(news.text.apply(lambda x: clean_text(x)))
        news = news.replace(to_replace='None', value=np.nan).dropna()
        news = news[news['text'].map(lambda x: len(x.strip())>50)]
        news['text'] = news.apply(lambda x: lemmatizer(x['text']), axis=1)
        news['text'] = news['text'].str.replace('-PRON-', '')

        print("**After Preprocessing** Dataframe Size: ")
        print(np.shape(news))
        print("="*50)
        print("Class Population: ")
        count_fake = news[news["target"]==0].count()["text"]
        count_true = news[news["target"]==1].count()["text"]
        print("Real (True) News Samples: "+str(count_true)+" / "+str(len(news)))
        print("Fake News Samples: "+str(count_fake)+" / "+str(len(news)))

        documents = [row.split() for row in news['text']]

        print("Word2Vec Building Vocabulary...")
        if(ds_count == 0):
            w2v_model = Word2Vec(documents,
                         min_count=MIN_WORD_COUNT,
                         window=TEXT_WINDOW,
                         size=EMBEDDING_DIM,
                         workers=6)
        else:
            w2v_model.build_vocab(documents,  update = True)
        
        print("Word2Vec Training Embedding Model...")
        w2v_model.train(documents, total_examples=len(documents), epochs=10)

        print("Word2Vec Dictionary Current Length: "+str(len(w2v_model.wv.vocab)))
        ds_count += 1

    print("Word2Vec Word Embedding Training Ended. ")
    w2v_model.init_sims(replace=True)
    w2v_model.save("saved_embeddings/w2v_fake_news.model")

def load_saved_embedding():
    w2v_model = Word2Vec.load(root_folder+
    "/embeddings/saved/w2v_fake_news.model")
    return w2v_model
