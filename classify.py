# basic array characteristics 
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
  
def oheClassify():
    data = pd.read_pickle('data/sentiment.pkl')
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
    return accuracy