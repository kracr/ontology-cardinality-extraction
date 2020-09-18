#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from sklearn.model_selection import train_test_split
Xtrain4, Xtest4, ytrain4, ytest4 = train_test_split(dataX4, dataY4, test_size=0.2, random_state=48)


# In[3]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

nb4= Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb4.fit(Xtrain4, ytrain4)

from sklearn.metrics import classification_report
ypredconcen = nb4.predict(Xtest4)



print('accuracy %s' % accuracy_score(ypredconcen, ytest4))
print(classification_report(ytest4, ypredconcen))


# In[4]:


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
sen = re.findall(r"\.(?P<sentence>.*?[0-9].*?)\.",drug1)  # if the sentence contain numeric value
print(sen)
"""
"""
import spacy
nlp = spacy.load('en')

tokens = nlp(drug1)
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

    


# In[5]:



y_pred_dose1 = nb4.predict(new)
print(y_pred_dose1)


with open("D:\\model\\Training data\\Results\\others\\Drugs.com\\Amphetamine\\AmphetamineConcentration.txt", 'a',encoding="utf-8") as f1:
 for n1,g1 in zip(new,y_pred_dose1):
    print(n1,g1,file=f1)
    


# In[9]:


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
     pattern=re.compile("(.)*((\d))(.)*")

     new1all=[]

     for element in newall:
       z = re.match(pattern, element)
       if z:
        new1all.append(element)
        #print(new1)
     newall=new1all

     y_pred_doseall = nb4.predict(newall)
     print(y_pred_doseall)


     with open('D:\\model\\drugs.com\\hasConcentrationResults1\\'+i, 'a',encoding="utf-8") as f1:
      for n1,g1 in zip(newall,y_pred_doseall):
       print(n1,g1,file=f1)
            

# do what you want with all these open files


# In[10]:



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
     pattern=re.compile("(.)*((\d))(.)*")

     new1all=[]

     for element in newall:
       z = re.match(pattern, element)
       if z:
        new1all.append(element)
        #print(new1)
     newall=new1all
    
     y_pred_doseall = nb4.predict(newall)
     print(y_pred_doseall)


     with open('D:\\model\\drugs.com\\pro_hasConcentrationResults1\\'+i, 'a',encoding="utf-8") as f1:
      for n1,g1 in zip(newall,y_pred_doseall):
       print(n1,g1,file=f1)
            

# do what you want with all these open files


# In[ ]:




