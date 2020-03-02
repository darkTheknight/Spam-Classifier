from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data = pd.read_csv('spambase.data').values
np.random.shuffle(data)

X = data[:, :48] #all rows and first 48 columns
Y = data[:, -1] #last column

#first 100 data set is for trainning
Xtrain = X[:-100,] 
Ytrain = Y[:-100,]

Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print ("Classification rate for NB:",model.score(Xtest, Ytest))


from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoostClassifier:",model.score(Xtest,Ytest))