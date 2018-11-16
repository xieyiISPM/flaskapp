# basic array characteristics 
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
  
# Creating array object 
def oheClassify():
    cols = ['polarity','id', 'date', 'query', 'user', 'tweet']
    data = pd.read_csv('data/sentiment.csv',names=cols, encoding='ISO-8859-1')
    #data = pd.read_pickle('data/sentiment.pkl')
    data=data.sample(frac=0.1,random_state=200)
    data = data.drop(['id', 'date', 'query', 'user'], axis=1)
    data.polarity = data.polarity.apply(lambda x: 1 if x==4 else x)
    # write the code here
    # train take 80% of data
    train = data.sample(frac=0.8, random_state=200)

    #this code tries to find that train = 80% and the rest of data would be test and dev data tables which is test_dev = 20%
    test_dev = data.loc[~data.index.isin(train.index)]

    # for test_dev data set, split half-half 
    #this code tries to find that test= 10% and dev = 10%
    test = test_dev.sample(frac=0.5, random_state=200)
    dev = test_dev.loc[~test_dev.index.isin(test.index),:]
    train.shape, dev.shape, test.shape
    # CountVectorizer is used in this case

    vectorizer = CountVectorizer(lowercase=True)
    X_train = vectorizer.fit_transform(train.tweet)
    y_train = train.polarity.as_matrix()

    X_dev = vectorizer.transform(dev.tweet)
    y_dev = dev.polarity.as_matrix()

    model = linear_model.LogisticRegression(penalty='l2')
    model.fit(X_train, y_train)
    accuracy = accuracy_score(model.predict(X_dev), y_dev) 
    return accuracy