#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk
from io import BytesIO
import codecs
import math
import nltk
from nltk.grammar import is_nonterminal
from nltk.tokenize import word_tokenize
import re
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

import numpy as np


# In[4]:



def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text


# In[5]:


def lower_case(tokens):
    normal_tokens = [w.lower() for w in tokens]     #normalised tokens
    return normal_tokens


# In[6]:


from rdflib import Graph
from rdflib import URIRef
g = Graph()
g.parse("D:\\model\\cardinal.owl")    # import owl file
aClass = URIRef("https://www.childhealthimprints.com/cardinalities/")

rdfType = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
#rdfType = URIRef("https://www.childhealthimprints.com/cardinalities/#hasPostmenstrualAge")

from rdflib.namespace import RDFS


#for triple in g.triples((None,rdfType,None)):
 #   print(triple)
    

subject=[]
predicate=[]
object=[]

for s,o,p in g.triples((None,None,None)):
    print(s.rsplit('#')[-1],o.rsplit('#')[-1],p.rsplit('#')[-1])
    



for subj, obj in g.subject_objects(predicate=rdfType):
    on=subj.rsplit('#')[-1]
    of=obj.rsplit('#')[-1]
#    print(on)
 #   print(of)

    
    
"""          
for g in on:
 search_keywords=''.join(on)
 print(search_keywords)
for g in of:
 search_keywords1=''.join(of)
 print(search_keywords1)
"""   
 


# In[7]:


def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text

trainpath="D:\\model\\Training data\\total.txt"                                                        
traindata=readfile(trainpath)
dataX=[]
dataY=[]
news=traindata.split("\n")
for line in news:
    line=line.strip()
    feature,label=line.split(":",2)
    dataX.append(feature)
    dataY.append(label)


print(dataX)
print(dataY)


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataX).toarray()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer()
datay=np.asarray(dataY)
#print(datax)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, datay, test_size=0.2, random_state=0)
#X_train1=X_train.reshape(3366,1)
#y_train=y_train.reshape(3366,1)



print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[9]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from sklearn.metrics import accuracy_score


import numpy as np
from sklearn import datasets
km = KMeans(n_clusters=2)
km.fit(X_train)
train=km.predict(X_train)
train=train.astype(str)
labels = km.labels_
print('accuracy %s' % accuracy_score(train, y_train))
print(y_train)
print(train)


# In[10]:


ypred = km.predict(X_test)
ypred=ypred.astype(str)


# In[11]:


from sklearn.metrics import classification_report

print('accuracy %s' % accuracy_score(ypred, y_test))

print(ypred)
print(y_test)


# In[12]:


from hmmlearn import hmm
import numpy as np
#HMM Model
gm = hmm.GaussianHMM(n_components=2)
gm.fit(X_train)
states = gm.predict(X_train)

ypred=gm.predict(X_test)

ypred=ypred.astype(str)

print('accuracy %s' % accuracy_score(ypred, y_test))


# In[13]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

knn = KNeighborsClassifier(n_neighbors=2) 
  
knn_model = NearestNeighbors(n_neighbors = 2, algorithm = 'auto').fit(X_train)
distances, indices = knn_model.kneighbors(X_test)

#print(distances) 
#print(indices)


"""
print("\nK Nearest Neighbors:")
for rank, index in enumerate(indices[0][:2], start = 1):
   print(str(rank) + " is", X[index])
"""
print('accuracy %s' % accuracy_score(indices, y_test))


# In[14]:


def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text

import codecs

trainpath1="D:\\model\\Training data\\hasDuration.txt"                                                        
traindata1=readfile(trainpath1)
dataX1=[]
dataY1=[]
news1=traindata1.split("\n")
for line in news1:
    line=line.strip()
    feature1,label1=line.split(":",2)
    dataX1.append(feature1)
    dataY1.append(label1)


# In[15]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X1 = vectorizer.fit_transform(dataX1).toarray()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer()
datay1=np.asarray(dataY1)


# In[16]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn import datasets
km1 = KMeans(n_clusters=2)
km1.fit(X1)
train1=km1.predict(X1)
train1=train1.astype(str)
labels = km1.labels_


# In[17]:


print('accuracy %s' % accuracy_score(train1, datay1))
print(datay1)
print(train1)


# In[18]:


from hmmlearn import hmm
import numpy as np
#HMM Model
gm = hmm.GaussianHMM(n_components=2)
gm.fit(X1)
states = gm.predict(X1)

ypred=gm.predict(X_test)

ypred=ypred.astype(str)

print('accuracy %s' % accuracy_score(ypred, datay1))


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

knn = KNeighborsClassifier(n_neighbors=2) 
  
knn_model = NearestNeighbors(n_neighbors = 2, algorithm = 'auto').fit(X1)
distances, indices = knn_model.kneighbors(X_test)

#print(distances) 
#print(indices)


"""
print("\nK Nearest Neighbors:")
for rank, index in enumerate(indices[0][:2], start = 1):
   print(str(rank) + " is", X[index])
"""
print('accuracy %s' % accuracy_score(indices, datay1))


# In[74]:


def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text

import codecs

trainpath2="D:\\model\\Training data\\hasVolume.txt"                                                        
traindata2=readfile(trainpath2)
dataX2=[]
dataY2=[]
news2=traindata2.split("\n")
for line1 in news2:
    line1=line1.strip()
    feature2,label2=line1.split(":",2)
    dataX2.append(feature2)
    dataY2.append(label2)


# In[75]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X2 = vectorizer.fit_transform(dataX2).toarray()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer()
datay2=np.asarray(dataY2)


# In[76]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn import datasets
km2 = KMeans(n_clusters=2)
km2.fit(X2)
train2=km2.predict(X2)
train2=train2.astype(str)
labels2 = km2.labels_


# In[77]:


print('accuracy %s' % accuracy_score(train2, dataY2))
print(train2)
print(train2)


# In[ ]:


from hmmlearn import hmm
import numpy as np
#HMM Model
gm = hmm.GaussianHMM(n_components=2)
gm.fit(X1)
states = gm.predict(X2)

ypred=gm.predict(X2)

ypred=ypred.astype(str)

print('accuracy %s' % accuracy_score(ypred, dataY2))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

knn = KNeighborsClassifier(n_neighbors=2) 
  
knn_model = NearestNeighbors(n_neighbors = 2, algorithm = 'auto').fit(X2)
distances, indices = knn_model.kneighbors(X2)

#print(distances) 
#print(indices)


"""
print("\nK Nearest Neighbors:")
for rank, index in enumerate(indices[0][:2], start = 1):
   print(str(rank) + " is", X[index])
"""
print('accuracy %s' % accuracy_score(indices, dataY2))


# In[78]:


def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text

import codecs

trainpath4="D:\\model\\Training data\\hasConcentration.txt"                                                        
traindata4=readfile(trainpath4)
dataX4=[]
dataY4=[]
news4=traindata4.split("\n")
for line4 in news4:
    line4=line4.strip()
    feature4,label4=line4.split(":",2)
    dataX4.append(feature4)
    dataY4.append(label4)


# In[79]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X4 = vectorizer.fit_transform(dataX4).toarray()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer()
datay4=np.asarray(dataY4)


# In[80]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X4, datay4, test_size=0.8, random_state=0)

import numpy as np
from sklearn import datasets
km4 = KMeans(n_clusters=2)
km4.fit(X_train)
train4=km4.predict(X_train)
train4=train4.astype(str)
labels4 = km4.labels_


# In[81]:


print('accuracy %s' % accuracy_score(train4, y_train))
print(train2)
print(datay4)
km4.fit(X_test)
test4=km4.predict(X_test)
test4=test4.astype(str)
print('accuracy %s' % accuracy_score(test4, y_test))


# In[ ]:





# In[ ]:


-


# In[82]:



from sklearn.model_selection import train_test_split
from pomegranate import NaiveBayes, NormalDistribution
model = NaiveBayes.from_samples(NormalDistribution, X4, datay4, verbose=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




