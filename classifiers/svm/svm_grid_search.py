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
from sklearn.svm import SVC
#from sklearn.linear_model import SGDClassifier
#from sklearn.model_selection import GridSearchCV
#from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
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
#data_fake = pd.read_csv('Fake_short.csv', encoding='utf-8')
#data_true = pd.read_csv('True_short.csv', encoding='utf-8')
data_fake = pd.read_csv('Fake_medium.csv', encoding='utf-8')
data_true = pd.read_csv('True_medium.csv', encoding='utf-8')
#fake_test = pd.read_csv('GettingReal/top10000_getting_real_about_fake_news.csv', encoding='utf-8')
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
fake_true_test['target'] = 'true'
fake_true_test.loc[fake_true_test['label'] == 'FAKE', ['target']] = 'fake'

news = pd.concat([data_true, data_fake]).reset_index(drop = True).sample(frac=1)
#print(news.head())
#print(fake_true_test.head())

porter_stemmer=PorterStemmer()
stopwords = set([w.lower() for w in stopwords.words('english')])
#lemma = nltk.download('wordnet')

stopwords_string = custom_preprocessor(' '.join(stopwords))
stopwords_normalized= set(stopwords_string.split())

TEST_SPLIT = 0.4

features=20000
ngram = (1, 3)
max_doc_freq=0.4
min_doc_freq=0.05
kernel='rbf'
#g=20
#c=2
print("Max Features: "+str(features))
print("N-gram range: "+str(ngram))
print("N-Gram Max Document Frequency: "+str(max_doc_freq))
print("N-Gram Min Document Frequency: "+str(min_doc_freq))
print("SVM Kernel: "+str(kernel))
#print("Gamma value: "+str(g))
#print("C value: "+str(c))

vectorizer = CountVectorizer(max_features=features, lowercase=True, analyzer='word',
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
vectorizer.fit(news['text'])
vectorized = vectorizer.transform(news['text'])
transformer = feature_extraction.text.TfidfTransformer()
transformed = transformer.fit_transform(vectorized)
#print("Vectorizer Dictionary :")
#print("Dictionary Length "+str(len(vectorizer.vocabulary_)))
#print(vectorizer.vocabulary_)
#print("Text")
#print(news.loc[0, 'text'])
#print("Vectorized")
#print(vectorized.shape)
#print(vectorized[0])
#print("Transformed: ")
#print(transformed.shape)

test_samples = int(TEST_SPLIT * np.shape(transformed)[0])
#print("test samples")
#print(test_samples)
x_train = transformed[:-test_samples]
y_train = news['target'][:-test_samples]
x_test = transformed[-test_samples:]
y_test = news['target'][-test_samples:]
#print(transformed[0])
#vectorize test set
x_fake_true_test = vectorizer.transform(fake_true_test['text'])
x_fake_true_test = transformer.transform(x_fake_true_test)

"""
C_range = np.logspace(-4, 4, base=10, num=9)
gamma_range = np.logspace(-4, 4, base=10, num=9)

print("c: ")
for c in C_range:
    print(c)

print("gamma: ")
for g in gamma_range:
    print(g)

param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(kernel=kernel), param_grid=param_grid, cv=cv, verbose=10)
grid.fit(x_train, y_train)


#The best SVM parameters are {'C': 10.0, 'gamma': 1.0} with a score of 0.99
print("The best SVM parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
"""

print("Cross Validating with best params...")
model = SVC(kernel=kernel, C=10.0, gamma=1.0)
model.fit(x_train, y_train)

scores = model_selection.cross_val_score(model, x_train, y_train, cv=10, verbose=10)
#print(scores)

prediction = model.predict(x_test)
prediction_fake_true = model.predict(x_fake_true_test)
print("Test Accuracy: ")
print(accuracy_score(y_test, prediction)*100)
print("Classification Report: ")
print(classification_report(y_test, prediction))

print("="*50)
print("Test Accuracy (2nd Dataset): ")
print(accuracy_score(fake_true_test['target'] , prediction_fake_true)*100)
print("Classification Report(2nd Dataset): ")
print(classification_report(fake_true_test['target'] , prediction_fake_true))
