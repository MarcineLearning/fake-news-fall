import numpy as np
import matplotlib.pyplot as plt

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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
VALIDATION_SPLIT = 0.1
MAX_LENGTH = 500

dataset=1
max_samples='NO'
max_samples=6500
cv_splits = 5
model_type_list=["KNN", "SVM", "LINEAR-SVM", 
"LOG-REG"]
embedding_type_list=["w2v", "glove", "tfidf-transformer"]

news = text.get_default_dataset(dataset, max_samples)
news, documents = text.preprocess_documents(news)

total_iterations=len(model_type_list)*len(embedding_type_list)
it_counter=1
for model_type in model_type_list:
    for embedding_type in embedding_type_list:
        w2v_model=0  
        print("="*50)
        print("iteration "+str(it_counter)+" of "+str(total_iterations))
        print("="*50)
        print("Model: "+model_type)
        print("Embedding/Features: "+embedding_type)
        print("Dataset: "+str(dataset))
        print("Samples: "+str(len(documents)))
        print("="*50)

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
            max_iter=2000)

        model.fit(x_train, y_train)

        scores = model_selection.cross_validate(model, x_train, y_train, 
            cv=cv_splits, verbose=1,
            scoring=('precision', 'recall', 'f1',  'accuracy'),
            return_train_score=True)
        print("Cross Validation Scores: ")
        metric_list = sorted(scores.keys())
        for m in metric_list:
            print("=>"+m)
            print(scores[m])

        prediction = model.predict(x_test)

        fpr, tpr, thresholds = roc_curve(y_test, prediction)
        area_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % area_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC '+str(model_type))
        plt.legend(loc="lower right")
        plt.savefig('output/roc_'+str(model_type)+'_'+str(embedding_type)+'_ds'+str(dataset)+'.png')
        plt.close()

        print("Test Accuracy: ")
        print(accuracy_score(y_test, prediction)*100)
        print("="*50)
        print("==== Classification Report ===========================")
        print(classification_report(y_test, prediction, labels=[0, 1]))
        print("="*50)   
        print("Confusion Matrix: ")
        print(confusion_matrix(y_test, prediction))
        print("="*50)
        print("="*50)   
        it_counter+=1
