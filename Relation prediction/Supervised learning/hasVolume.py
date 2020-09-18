#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


from sklearn.model_selection import train_test_split
Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split(dataX2, dataY2, test_size=0.2, random_state=48)


# In[4]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

nb3= Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb3.fit(Xtrain2, ytrain2)

from sklearn.metrics import classification_report
ypredvolume = nb3.predict(Xtest2)



print('accuracy %s' % accuracy_score(ypredvolume, ytest2))
print(classification_report(ytest2, ypredvolume))


# In[5]:


import re
import codecs
from nltk.tokenize import sent_tokenize


def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text


otherspath="D:\\model\\Training data\\Results\\others\\Drugs.com\\Amphetamine\\Amphetamine.txt"                                                        
drug1=readfile(otherspath)

#sen = re.findall(r"\.(?P<sentence>.*?[0-9].*?)\.",drug1)  # if the sentence contain numeric value
#sen=str(sen)
#print(sen)
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


# In[6]:


y_pred_volume = nb3.predict(new)
print(y_pred_volume)


with open('D:\\model\\Training data\\Results\\others\\Drugs.com\\Amphetamine\\AmphetamineVolume.txt', 'a',encoding="utf-8") as f1:
 for n1,g1 in zip(new,y_pred_volume):
    print(n1,g1,file=f1)
    


# In[7]:


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
     pattern=re.compile("(.)*((\d))(.)*")

     new1all=[]

     for element in newall:
       z = re.match(pattern, element)
       if z:
        new1all.append(element)
        #print(new1)
     newall=new1all
     y_pred_doseall = nb3.predict(newall)
     print(y_pred_doseall)


     with open('D:\\model\\drugs.com\\hasVolumeResults1\\'+i, 'a',encoding="utf-8") as f1:
      for n1,g1 in zip(newall,y_pred_doseall):
       print(n1,g1,file=f1)
            

# do what you want with all these open files


# In[8]:



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
     y_pred_doseall = nb3.predict(newall)
     print(y_pred_doseall)


     with open('D:\\model\\drugs.com\\pro_hasVolumeResults1\\'+i, 'a',encoding="utf-8") as f1:
      for n1,g1 in zip(newall,y_pred_doseall):
       print(n1,g1,file=f1)
            

# do what you want with all these open files


# In[ ]:




