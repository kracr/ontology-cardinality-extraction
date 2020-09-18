#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp("Amikacin has minimum dosage of 100mg")

for tok in doc:
  print(tok.text, "...", tok.dep_)


# In[4]:


import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import codecs
def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text


# In[7]:


trainpath="D:\\model\\Training data\\total1333.txt"                                                        
"""traindata=readfile(trainpath)
dataX=[]
dataY=[]
news=traindata.split("\n")
for line in news:
    line=line.strip()
    feature,label=line.split(":",2)
    dataX.append(feature)
    dataY.append(label)

#print(neofax)
chil=str(dataX)
from nltk.tokenize import sent_tokenize
sent=(sent_tokenize(chil))
print(sent)
"""
trainpath="D:\\model\\Training data\\total1333.txt"                                                        
traindata=readfile(trainpath)
news=traindata.split("\n")
from nltk.tokenize import sent_tokenize
sent=(sent_tokenize(str(news)))
#print(sent)


# In[8]:


def remove_punctuations(tokens):
    table = str.maketrans('', '', string.punctuation) #removed punctuations from the tokens
    ptokens = [w.translate(table) for w in tokens]     #bible-believer: biblebeliever
    return ptokens
def remove_blank_tokens(tokens):
    non_blank_tokens = [s for s in tokens if s]
    return non_blank_tokens
from nltk.stem import WordNetLemmatizer


# In[9]:


from nltk.corpus import stopwords

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    words = [t for t in tokens if not t in stop_words]
    return words

def lametizer(tokens):
    lemmatizer = WordNetLemmatizer() 
    lematized_tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return lematized_tokens


# In[10]:


from nltk.stem import PorterStemmer 

def stemmer(tokens):
    pr = PorterStemmer() 
    stemmed_tokens = [pr.stem(t) for t in tokens]
    return stemmed_tokens


# In[11]:


from nltk.tokenize import sent_tokenize, word_tokenize
import string

def fun(sentence):
  tokens = word_tokenize(sentence)
  ptokens=remove_punctuations(tokens)
  btokens=remove_blank_tokens(ptokens)
  simple_tokens=remove_stopwords(btokens)
  lame_tokens=lametizer(simple_tokens)
  stemmed_tokens = stemmer(lame_tokens)
  fstemmed_tokens=remove_blank_tokens(stemmed_tokens)
  return(fstemmed_tokens)

for sentence in sent:
    sentence=fun(sentence)
    #print(sentence)
    
for sentence in sent:
    for word in sentence:
     if(sentence=='.'):
        sent.remove(word)
     if(sentence==','):
        sent.remove(word)
     if(sentence==':'):
        sent.remove(word)
    #print(sentence)


# In[12]:


new_sent=[]
unique_list=[]
for sente in sent:
    sente=sente.strip()
    for sentee in sente:
        if(sentee.isdigit()):
            new_sent.append(sente)
#print(new_sent)

list_set = set(new_sent) 
unique_list = (list(list_set)) 




# In[13]:


new=', '.join(map(str, unique_list))
#print(new)


# In[14]:


import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
stop_words = set(stopwords.words('english')) 


# In[15]:


import spacy
new_sent=new
new_sent=new_sent.split()
#print(new_sent)

new=' '.join(map(str, new_sent))
#print(new)

from random import sample
import random

new_sent=new
new_sent.strip()

#print(new_sent)
taggedfinal={}

new_sent=new_sent.split(',')
for sent in new_sent:
  
  doc = nlp(sent)
#  for tok in doc:
      #print(tok.text, "...", tok.dep_)
        


# In[17]:


def get_entities(sent):
  ## chunk 1
  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence

  prefix = ""
  modifier = ""

  
  for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
      
      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text
      
      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""      

      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text
        
      ## chunk 5  
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text

  return [ent1.strip(), ent2.strip()]


# In[18]:


get_entities("The modern rules of many football codes were formulated during the mid or late 19 century")


# In[19]:


entity_pairs = []
quantifier=[]

for i in tqdm(new_sent):
  entity_pairs.append(get_entities(i))


# In[21]:


#print(entity_pairs)


# In[16]:


def get_relation(sent):

  doc = nlp(sent)

  # Matcher class object 
  matcher = Matcher(nlp.vocab)

  pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

  matcher.add("matching_1", None, pattern) 

  matches = matcher(doc)
  k = len(matches) - 1

  span = doc[matches[k][1]:matches[k][2]] 

  return(span.text)


# In[ ]:





# In[18]:


relations = [get_relation(i) for i in tqdm(new_sent)]


# In[19]:


#print(relations)


# In[20]:


pd.Series(relations).value_counts()


# In[27]:


def get_number1(sent):
  gt=sent
  gt=gt.split(' ')
  for gtoo in gt:
   gtoo=gtoo.split(' ')
   for gtoo1 in range(len(gtoo)):
     if((gtoo[gtoo1]).isdigit()):
       print("mmmmmm",gtoo1)
  return gtoo1,gtoo[gtoo1]


# In[28]:


def get_number(sent):
   sent=sent.strip(" ")
   sent=sent.split(" ")
   print(sent)
   for pos in range(len(sent)):
        #print(pos)
        try:
          if (sent[pos].isnumeric()):
           #print("position",pos,"value",sent[pos])
           return pos,sent[pos]
          else:
           return Nil,Nil      
        except:
           pass

    
"""    
  gt=sent
 # gt=gt.split(" ")
  print(gt)
  for gtoo1 in range(len(gt)):
    print(gt[gtoo1])
    if((gt[gtoo1]).isnumeric()):
       #print("mmmmmm",gtoo1)
       print(gtoo1,gt[gtoo1])
    return gtoo1,gt[gtoo1]
"""


# In[1]:


for i in tqdm(new_sent):
     #print(i)
     try:
       position, number=get_number(i)
      # print("position:",position,"number:",number)
     except:
       pass


# In[30]:


def get_quantifier(sent):
  

  
  neww=[]
  
  #############################################################
  docu=nlp(sent)
  for tok in range(len(docu)):
    if ((docu[tok].dep_ == 'amod') & (docu[tok].text=='less')):
              if docu[tok-1].dep_ == 'neg':
                   quantifier='Minimum'
                   print(quantifier,docu[tok-1].text,docu[tok].text)
                   return quantifier
              if docu[tok-1].dep_ !='neg':  
                   quantifier='Maximum'
                   print(quantifier,docu[tok].text)
                   return quantifier

                    


    if ((docu[tok].dep_ == 'amod') & (docu[tok].text=='more')):
               if docu[tok-1].dep_ == 'neg':
                   quantifier='Maximum'
                   print(quantifier,docu[tok-1].text,docu[tok].text)
                   return quantifier

               if docu[tok-1].dep_ !='neg':  
                   quantifier='Minimum'
                   print(quantifier,docu[tok].text)
                   return quantifier



    if(docu[tok].dep_=='acomp'):
         if docu[tok].text=='maximum' or docu[tok].text=='Maximum':
              if docu[tok-1].dep_ == 'neg':
                   quantifier='Minimum'
                   print(quantifier,docu[tok-1].text,docu[tok].text)
                   return quantifier

              if docu[tok-1].dep_ !='neg':  
                   quantifier='Maximum'
                   print(' Maximum',docu[tok].text)
                   return quantifier

  
    if(docu[tok].dep_=='acomp'):
         if docu[tok].text=='minimum':
              if docu[tok-1].dep_ == 'neg':
                   quantifier='Maximum'
                   print(quantifier,docu[tok-1].text,docu[tok].text)
                   return quantifier

              if docu[tok-1].dep_ !='neg':  
                   quantifier='Minimum'
                   print(' Minimum',docu[tok].text)
                   return quantifier

  


    if(docu[tok].dep_=='acomp'):
         if docu[tok].text=='greater':
              if docu[tok-1].dep_ == 'neg':
                   quantifier='Maximum'
                   print(quantifier,docu[tok-1].text,docu[tok].text)
                   return quantifier

              if docu[tok-1].dep_ !='neg':  
                   quantifier='Minimum'
                   print(' Minimum',docu[tok].text)
                   return quantifier

  
    if(docu[tok].dep_=='acomp'):
         if docu[tok].text=='equal':
              if docu[tok-1].dep_ == 'neg':
                   quantifier='Not equal to'
                   print(quantifier,docu[tok-1].text,docu[tok].text)
                   return quantifier

              if docu[tok-1].dep_ !='neg':  
                   quantifier='Exact'
                   print(' Exact',docu[tok].text)
                   return quantifier

  
    if(docu[tok].dep_=='advmod'):
         if docu[tok].text=='exactly':
              if docu[tok-1].dep_ == 'neg':
                   quantifier='Not equal quantifier'
                   print(quantifier,docu[tok-1].text,docu[tok].text)
                   return quantifier

              if docu[tok-1].dep_ !='neg':  
                   quantifier='Exact quantifier'
                   print(' Exact quantifier',docu[tok].text)
                   return quantifier

             
              
      
    if(docu[tok].dep_=='acomp'):
         if docu[tok].text=='exact':
              if docu[tok-1].dep_ == 'neg':
                   quantifier='Not equal'
                   print(quantifier,docu[tok-1].text,docu[tok].text)
                   return quantifier

              if docu[tok-1].dep_ !='neg':  
                   quantifier='Exact'
                   print(' Exact',docu[tok].text)
                   return quantifier

             
    
    ###########################################################


# In[31]:


import spacy
nlp = spacy.load('en_core_web_sm')

maximum=['Maximum','maximum','less than','no greater than','not to exceed','no more than']
minimum=['Minimum','greater than','atleast','minimum','no lesser than','more than']
equal=['equal to','exact']

tt='My age is not exactly 7'
quantifier=get_quantifier(tt)
print(quantifier)
tt1=nlp(tt)
for tok1 in tt1:
      print(tok1.text, "...", tok1.dep_)



# In[32]:


quantifier=[]
number=[]
for i in tqdm(new_sent):
    #print(i)
    quantifier.append(get_quantifier(i))
    try:
     position, num=get_number(i)
     number.append(num)

    except:
     pass

print(quantifier)
print(number)


# In[45]:


sub_obj=[]
rel=[]
quant=[]
card=[]
posi_numb=[]


for i in new_sent:
  try:
    sub_obj.append(get_entities(i))
    rel.append(get_relation(i)) 
    posi_numb=get_number(i)
    quant.append(get_quantifier(i))
  except:
    pass

for i in new_sent:
 try:
  for i,j in sub_obj,posi_numb:
    if(i[0]!=None & i[1]!=None & j[0]!=None & j[1]!=None):
        subj=i[0]
        obj=i[1]
        position1=j[0]
        number1=j[1]
        
        #print(subj,obj,position1,number1)
 except:
    pass
#subj = [i[0] for i in sub_obj]
#obj = [i[1] for i in sub_obj]
#position1=[i[0] for i in posi_numb]
#number1=[i[1]for i in posi_numb]
#print(subj,obj,position1,number1)

    
    
    
    


# In[46]:


# extract subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

cardinal=[i for i in quantifier]

number=[i for i in number]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

#kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations,'cardinal':cardinal, 'number':number})
#df = pd.DataFrame.from_dict(kg_df, orient='index')
#df.transpose()
 

print(source,target,relations,cardinal,number)


# In[ ]:





# In[47]:


G=nx.from_pandas_edgelist(kg_df, "source", "target",
                          edge_attr=True, create_using=nx.MultiDiGraph())


# In[48]:


plt.figure(figsize=(12,12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


# In[51]:


G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="consider"], "source", "target",
                          edge_attr=True, create_using=nx.MultiDiGraph())

#plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
#plt.show()


# In[ ]:





# In[52]:


piano_class_doc = nlp(str(new_sent))
for ent in piano_class_doc.ents:
   print(ent.text, ent.start_char, ent.end_char,
           ent.label_, spacy.explain(ent.label_))


# In[60]:


import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
doc = nlp("San Francisco considers banning sidewalk delivery robots")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

doc = nlp("FB is hiring a new VP of global policy")
doc.ents = [Span(doc, 0, 1, label="ORG")]
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)


# In[61]:


import spacy
import random

nlp = spacy.load("en_core_web_sm")
train_data = [("Uber blew through $1 million", {"entities": [(0, 4, "ORG")]})]

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for i in range(10):
        random.shuffle(train_data)
        for text, annotations in train_data:
            nlp.update([text], [annotations], sgd=optimizer)
nlp.to_disk("/model")


# In[ ]:


from spacy import displacy

doc_dep = nlp("This is a sentence.")
displacy.serve(doc_dep, style="dep")

doc_ent = nlp("When Sebastian Thrun started working on self-driving cars at Google "
              "in 2007, few people outside of the company took him seriously.")
displacy.serve(doc_ent, style="ent")


# In[ ]:




