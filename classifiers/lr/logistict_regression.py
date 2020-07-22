#Pandas
import pandas as pd
#Numpy
import numpy as np
#Seaborn
import seaborn as sns
#Matplotlib
import matplotlib.pyplot as plt
#SKLEARN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
#NLTK
import nltk
from nltk.corpus import stopwords 
from nltk.util import ngrams
from nltk.stem import PorterStemmer
 
#SPACY
import spacy
from spacy import displacy
from spacy.util import minibatch, compounding
#STD LIB
import string
import collections
import re
import copy

"""
Data
"""
#data_fake = pd.read_csv('Fake.csv', encoding='utf-8')
#data_true = pd.read_csv('True.csv', encoding='utf-8')
data_fake = pd.read_csv('Fake_short.csv', encoding='utf-8')
data_true = pd.read_csv('True_short.csv', encoding='utf-8')
fake_test = pd.read_csv('GettingReal/top10000_getting_real_about_fake_news.csv', encoding='utf-8')
fake_true_test = pd.read_csv('RealOrFake/top10000_fake_or_real_news.csv', encoding='utf-8')

#file_data_fake = pd.read_csv('Fake.csv', encoding='utf-8')
#file_data_true = pd.read_csv('True.csv', encoding='utf-8')

#data_true= pd.DataFrame(file_data_true.loc[file_data_true['subject'] == "politicsNews"])
#data_fake= pd.DataFrame(file_data_fake.loc[file_data_fake['subject'] == "politics"])
#data_true.reset_index(drop = True)
#data_fake.reset_index(drop = True)

#sns.countplot(data_fake['subject']);
#sns.countplot(data_true['subject']);


def unique(text):
    for word in text:
        text = text.split()
        text = set(text)
        return text
    
def removeBlackListed(row_text, blacklist):
    word_list = row_text.split()
    result=[]
    for word in word_list:
        if(len(word)>1):
            if word not in blacklist:
                result.append(word)
    return " ".join(result)

def my_cool_preprocessor(text):
    text=text.lower() 
    text=re.sub("\\W"," ",text) # remove special chars
    text=re.sub("\\s+(in|the|all|for|and|on)\\s+"," _connector_ ",text) # normalize certain words
    
    # stem words
    words=re.split("\\s+",text)
    stemmed_words=[porter_stemmer.stem(word=word) for word in words]
    return ' '.join(stemmed_words)    

def custom_preprocessor(text):
    #remove punctuation
    text = text.lower()
    #text = "".join([char for char in word if char not in string.punctuation])
    text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub('[0-9]+', '', text)
    #remove digits
    text = ''.join([i for i in text if not i.isdigit()])
    #remove URLs
    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'',text)
    #remove HTML tags
    html=re.compile(r'<.*?>')
    text = html.sub(r'',text)
    # stem words
    words=re.split("\\s+",text)
    #stemmed_words=[porter_stemmer.stem(word=word) for word in words]
    #return ' '.join(stemmed_words)    
    return ' '.join(words)    


data_fake['target'] = 'fake'
data_true['target'] = 'true'
fake_test['target'] = 'fake'
fake_true_test['target'] = 'true'
fake_true_test.loc[fake_true_test['label'] == 'FAKE', ['target']] = 'fake'

news = pd.concat([data_true, data_fake]).reset_index(drop = True)
print(news.head())
print(fake_true_test.head())

porter_stemmer=PorterStemmer()
stopwords = set([w.lower() for w in stopwords.words('english')])
#lemma = nltk.download('wordnet')

stopwords_string = custom_preprocessor(' '.join(stopwords))
stopwords_normalized= set(stopwords_string.split())

#x_train,x_test,y_train,y_test = model_selection.train_test_split(news['text'], news.target, test_size=0.3)

#vectorizer = CountVectorizer()

# con LogisticRegression
# ngram 2,4
# RealOrFake Dataset accuracy: 69.82%
# Pure Fake Dataset accuracy: 84.6%
# ngram 2,3
# RealOrFake Dataset accuracy: 67.99%
# Pure Fake Dataset accuracy: 86.5%
# ngram 2,3 con Dataset FULL
# RealOrFake Dataset accuracy: 54.27%
# Pure Fake Dataset accuracy: 86.7%

# RealOrFake Dataset accuracy: 70.12%
# Pure Fake Dataset accuracy: 80.5%
vectorizer = CountVectorizer(max_features=20000, lowercase=True, analyzer='word', 
            stop_words= 'english',ngram_range=(2, 2), max_df=0.05)
            
# con LinearSVC             
# RealOrFake Dataset accuracy 70.12% con ngram 2,4
# ngram 2,3
# RealOrFake Dataset accuracy: 68.9%
# Pure Fake Dataset accuracy: 80.2%
# ngram 2,3 Dataset FULL
# RealOrFake Dataset accuracy: 55.49%
# Pure Fake Dataset accuracy: 87.0%

#vectorizer = CountVectorizer(max_features=8000, lowercase=True, analyzer='word',
            #preprocessor=custom_preprocessor                 
            #stop_words= stopwords_normalized
            #stop_words= 'english',
            #ngram_range=(2, 3), max_df=0.05)

vectorized = vectorizer.fit_transform(news['text'])
transformer = feature_extraction.text.TfidfTransformer()
transformed = transformer.fit_transform(vectorized)
#vectorize test set
x_fake_test = vectorizer.transform(fake_test['text'])
x_fake_test = transformer.transform(x_fake_test)
x_fake_true_test = vectorizer.transform(fake_true_test['text'])
x_fake_true_test = transformer.transform(x_fake_true_test)


# LOGISTIC REGRESSION
"""
[CV]  ................................................................
[CV] .................................... , score=0.929, total=  52.5s
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:   52.5s remaining:    0.0s
[CV]  ................................................................
[CV] .................................... , score=0.938, total= 1.2min
[Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:  2.0min remaining:    0.0s
[CV]  ................................................................
[CV] .................................... , score=0.973, total= 1.2min
[Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:  3.2min remaining:    0.0s
[Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:  3.2min finished
RealOrFake Dataset accuracy: 55.18%
Pure Fake Dataset accuracy: 86.8%
"""

model = linear_model.LogisticRegression(solver = 'saga', 
        class_weight='balanced', 
        penalty='none',
        max_iter=1500,
        random_state=150)

features = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 11000, 12000, 13000, 14000, 15000]
ngram_ranges=[(1, 2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,3) ,(3,4)]
max_dfrs =[0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.10]
#model=SVC(kernel='sigmoid', gamma=20, C=2)
#model=SVC(kernel='rbf', gamma=0.1, C=3)


for f in features:
    for ng in ngram_ranges:
        for freq in max_dfrs:
            print("Max Features: "+str(f))
            print("Ngram Range: "+str(ng))
            print("Max document frequence: "+str(freq))
            vectorizer = CountVectorizer(max_features=f, lowercase=True, analyzer='word',
                        ##preprocessor=custom_preprocessor                 
                        ##stop_words= stopwords_normalized
                        stop_words= 'english',
                        ngram_range=ng, max_df=freq)

            vectorized = vectorizer.fit_transform(news['text'])
            transformer = feature_extraction.text.TfidfTransformer()
            transformed = transformer.fit_transform(vectorized)
            #vectorize test set
            #x_fake_test = vectorizer.transform(fake_test['text'])
            #x_fake_test = transformer.transform(x_fake_test)
            x_fake_true_test = vectorizer.transform(fake_true_test['text'])
            x_fake_true_test = transformer.transform(x_fake_true_test)

            #model.fit(transformed, y_train)
            model.fit(transformed, news.target)
            prediction_fake_true = model.predict(x_fake_true_test)
            #prediction_fake = model.predict(x_fake_test)
            prediction = model.predict(transformed)

            scores = model_selection.cross_val_score(model, transformed, news.target, cv=3, verbose=10)
            #print(scores)

            #print(accuracy_score(y_test, prediction)*100)
            #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            #print(classification_report(y_test, prediction))


            
            print("RealOrFake Dataset accuracy: {}%".format(round(accuracy_score(fake_true_test.target, prediction_fake_true)*100,2)))
            print(" ")
            #print(confusion_matrix(fake_true_test.target, prediction_fake_true))
            #print(classification_report(fake_true_test.target, prediction_fake_true))

            #print(" ")
            #print("Pure Fake Dataset accuracy: {}%".format(round(accuracy_score(fake_test.target, prediction_fake)*100,2)))
            #print(confusion_matrix(fake_test.target, prediction_fake))
            #print(classification_report(fake_test.target, prediction_fake))

