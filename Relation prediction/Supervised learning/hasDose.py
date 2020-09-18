#!/usr/bin/env python
# coding: utf-8

# In[1]:




def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text

import codecs

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

#print(dataY)


# In[2]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(dataX, dataY, test_size=0.2, random_state=48)


# In[3]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(Xtrain, ytrain)

from sklearn.metrics import classification_report
ypred = nb.predict(Xtest)
#ypred=int(ypred)


print('accuracy %s' % accuracy_score(ypred, ytest))
print(classification_report(ytest, ypred))


# In[6]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(Xtrain).toarray()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer()
y_train=np.asarray(ytrain)
#b = X1[1].transpose().copy()
#b.resize((99), refcheck=False)
#b = b.transpose()

#b=b.astype(float)


X_test = vectorizer.fit_transform(Xtest).toarray()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer()
y_test=np.asarray(ytest)



from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier


rf =  RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
gg=rf.predict(X_test)
print(confusion_matrix(y_test, gg))
score = accuracy_score(y_test, gg)
print("Random forest score",score)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
knear=classifier.predict(X_test) # 0:Overcast, 2:Mild
scorek = accuracy_score(y_test, knear)
print(confusion_matrix(y_test, knear))
print("KNeighborsClassifier score",scorek)


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
  
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
grid.fit(X_train, y_train)
svmc=grid.predict(X_test) # 0:Overcast, 2:Mild
scorek = accuracy_score(y_test, svmc)
print(confusion_matrix(y_test, knear))
print("SVM linear score",scorek)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


import re
import codecs
from nltk.tokenize import sent_tokenize


def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text


otherspath="D:\\model\\Training data\\Results\\others\\Drugs.com\\Amphetamine\\Amphetamine.txt"                                                        
drug1=readfile(otherspath)

"""
import spacy
nlp = spacy.load('en')

tokens = nlp((drug1))
new=[]


for sent in tokens.sents:
    new.append(sent.string.strip())
    
   """


#new=sent_tokenize(drug1)
new= '\n'.join([x for x in drug1.split("\n") if x.strip()!=''])

new=new.split('\n')
pattern=re.compile("(.)*((\d))(.)*")

new1=[]

for element in new:
    z = re.match(pattern, element)
    if z:
     new1.append(element)
print(new1)
new=new1


# In[22]:


"""
for sente in (new):
 for word in sente:
     if (word.isnumeric()):            
          new.append(sente.strip())

    
#drug11=sent_tokenize(drug1)
"""


# In[23]:



y_pred_dose1 = nb.predict(new)
print(y_pred_dose1)


with open('D:\\model\\Training data\\Results\\others\\Drugs.com\\Amphetamine\\AmphetamineDose.txt', 'a',encoding="utf-8") as f1:
 for n1,g1 in zip(new,y_pred_dose1):
    print(n1,g1,file=f1)
    


# In[24]:



import re
import codecs
from nltk.tokenize import sent_tokenize


def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text



import os
file_content =[]
dir = "D:\\model\\drugs.com\\normal\\"
files = []
for i in os.listdir(dir):
     drugall=[]
     newall=[]
     y_pred_doseall=[]
        
     filepath=dir+i
     drugall=readfile(filepath)
     newall= '\n'.join([x for x in drugall.split("\n") if x.strip()!=''])

     newall=newall.split('\n')
     """ 
    pattern=re.compile("(.)*((\d))(.)*")

     new1all=[]

     for element in newall:
       z = re.match(pattern, element)
       if z:
        new1all.append(element)
        #print(new1)
     newall=new1all
     """
     y_pred_doseall = nb.predict(newall)
    
     print(y_pred_doseall)


     with open('D:\\model\\drugs.com\\hasDoseResults\\'+i, 'a',encoding="utf-8") as f1:
      for n1,g1 in zip(newall,y_pred_doseall):
       print(n1,g1,file=f1)
            

# do what you want with all these open files


# In[25]:



import re
import codecs
from nltk.tokenize import sent_tokenize


def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text



import os
file_content =[]
dir = "D:\\model\\drugs.com\\pro\\"
files = []
for i in os.listdir(dir):
     drugall=[]
     newall=[]
     y_pred_doseall=[]
        
     filepath=dir+i
     drugall=readfile(filepath)
     newall= '\n'.join([x for x in drugall.split("\n") if x.strip()!=''])
    
     newall=newall.split('\n')
     """
     pattern=re.compile("(.)*((\d))(.)*")

     new1all=[]

     for element in newall:
       z = re.match(pattern, element)
       if z:
        new1all.append(element)
        #print(new1)
     newall=new1all
     """
     y_pred_doseall = nb.predict(newall)
     print(y_pred_doseall.count('1'))


   #  with open('D:\\model\\drugs.com\\pro_hasDoseResults\\'+i, 'a',encoding="utf-8") as f1:
    #  for n1,g1 in zip(newall,y_pred_doseall):
     #  print(n1,g1,file=f1)
            

# do what you want with all these open files


# In[ ]:





# In[ ]:




