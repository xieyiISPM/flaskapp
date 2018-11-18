# basic array characteristics 
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score

from sklearn import metrics

  
def oheClassify():
    data = pd.read_pickle('data/sentiment.pkl')
    data.polarity = data.polarity.map(lambda x: 1 if x==4 else 0)
    train = data[:5000]
    test =data[5000:]
    cvX = CountVectorizer(token_pattern="\\w+", lowercase=True)
    cvX.fit(data.tweet)
    X_train = cvX.transform(train.tweet)
    y_train = train.polarity.as_matrix()

    X_test = cvX.transform(test.tweet)
    y_test = test.polarity.as_matrix()

    model = linear_model.LogisticRegression(penalty='l2')
    model.fit(X_train, y_train)
    accuracy = accuracy_score(model.predict(X_test), y_test) 
    #generate ROC plot
    import matplotlib.pyplot as plt
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr_ohe,tpr_ohe, _=metrics.roc_curve(y_test,y_pred_proba)
    auc_ohe =metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr_ohe, tpr_ohe, label='w2v, auc='+ str(auc_ohe), color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
   
    plt.savefig('static/ohe_plot.png')
    plt.clf()
    plt.close("all")
    return accuracy

def w2vClassify():
    data = pd.read_pickle('data/sentiment.pkl')
    data.polarity = data.polarity.map(lambda x: 1 if x==4 else 0)
    train = data[:5000]
    test =data[5000:]
    model = linear_model.LogisticRegression(penalty='l2')
    X_train_w2v = vecfun(train)
    y_train_w2v = train.polarity.as_matrix()
    X_test = vecfun(test)
    model.fit(X_train_w2v, y_train_w2v)
    X_test_w2v = X_test
    y_test_w2v = test.polarity.as_matrix()
    w2vaccuracy = accuracy_score(model.predict(X_test_w2v), y_test_w2v) 
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr,tpr, _=metrics.roc_curve(y_test_w2v,y_pred_proba)
    auc =metrics.roc_auc_score(y_test_w2v, y_pred_proba)
    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr, label='w2v, auc='+ str(auc), color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
   
    plt.savefig('static/w2v_plot.png')
    plt.clf()
    plt.close('all')
    return w2vaccuracy

def vecfun(data):
    vec2=[]
    i= 0
    for x in data.w2v.values:
        if type(x)!= np.float64:
            vec=[]
            for i in range(len(x)):
                 vec.append(x[i])
        vec2.append(vec)
    return np.array(vec2)


