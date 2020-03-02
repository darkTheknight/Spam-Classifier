from __future__ import print_function, division
from future.utils import iteritems
from builtins import range



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

#load the data
df = pd.read_csv('spam.csv',encoding='ISO-8859-1' )


#drop unnecessary columns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

#rename columns to something better
df.columns = ['labels', 'data']

#create blabels new colomn
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].values

count_vectorizer = CountVectorizer(decode_error='ignore')
X = count_vectorizer.fit_transform(df['data'])

#split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

#create the model
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:",model.score(Xtrain, Ytrain))
print("test score:",model.score(Xtest, Ytest))


def visualize(label):
  words = ''
  for msg in df[df['labels'] == label]['data']:
    msg = msg.lower()
    words += msg + ' '
  wordcloud = WordCloud(width=600, height=400).generate(words)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()

visualize('spam')
visualize('ham')


#create new column prediction and set value for our model
df['prediction'] = model.predict(X)

#these are have to be spam
sneaky_spam = df[(df['prediction'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
  print(msg)

#these are not supposed to be spam

not_actually_spam = df[(df['prediction'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_actually_spam:
  print(msg)

not_actually_spam = df[(df['prediction'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_actually_spam:
	print(msg)






