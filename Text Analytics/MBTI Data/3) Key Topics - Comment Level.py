
# coding: utf-8

# In[ ]:


'3) Key Topic Modeling'

'''
This file takes the data that was prepared in the 1) Data Prep and uses it to extract the top 20 
key topics, by MBTI type, for the comments. 

The files does the following:
    - Loads libraries and User Defined Functions
    - Loads the Data and Creates a list of MBTI types
    - Loops over the comments data by MBTI and extracts the top 20 topics and prints them to the screen
'''


# In[ ]:


#Import Libraries
import numpy as np
import pandas as pd
import pickle

from gensim import corpora, models
from normalization import normalize_corpus


# In[ ]:


#Load User Defined Functions
def print_topics_gensim(topic_model, total_topics=1,
                        weight_threshold=0.0001,
                        display_weights=False,
                        num_terms=None):
    
    for index in range(total_topics):
        topic = topic_model.show_topic(index)
        topic = [(word, round(wt,2)) 
                 for word, wt in topic 
                 if abs(wt) >= weight_threshold]
        if display_weights:
            print('Topic #'+str(index+1)+' with weights')
            print(topic[:num_terms] if num_terms else topic)
        else:
            print('Topic #'+str(index+1)+' without weights')
            tw = [term for term, wt in topic]
            print(tw[:num_terms] if num_terms else tw)
        print()


# In[ ]:


#Load the cleaned MBTI data
cleaned_mbti_token_cmtlvl = pd.read_pickle("cleaned_mbti_token_cmtlvl.pkl")
print(cleaned_mbti_token_cmtlvl.head())

mbti_list = cleaned_mbti_token_cmtlvl.iloc[:,3].values.tolist()
mbti_list = list(set(mbti_list))

print(mbti_list)


# In[ ]:


#For each MBTI Type in the data, extract the top 20 topics 

total_topics = 20 #Number of topics

#Define the words to be removed from the lists
wordlist = ['url','infj','intj','infp','intp','enfj','entj','enfp','entp','isfj',
                'istj','isfp','istp','esfj','estj','esfp','estp','tapatalk']

for mbti in mbti_list:
    print('-----------------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------------')
    print(mbti)
    
    #Subset the data
    subset = cleaned_mbti_token_cmtlvl.loc[cleaned_mbti_token_cmtlvl['type'] == mbti]
    print(subset.shape)
    print()
    
    #Transform the data to list form
    features = subset.iloc[:,5:].values.tolist()
    print(features[0:2])
    print()

    #Remove common/superfuerlous words from the lists
    feature_none = []
    for x in features:
        y = list(filter(None.__ne__, x))
        z = [z for z in y if z not in wordlist]    
        feature_none.append(z)
        
    #Create a dictionary of the words
    dictionary = corpora.Dictionary(feature_none)
    print(len(dictionary.token2id))

    #Transform the document to a BOW
    corpus = [dictionary.doc2bow(text, allow_update=True) for text in feature_none]
    #print(corpus[:2])

    #Transform to TFIDF
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    #Extract top topics using Latent Semantic Indexing
    lsi = models.LsiModel(corpus_tfidf, 
                          id2word=dictionary, 
                          num_topics=total_topics)

    #Print the top topics
    print_topics_gensim(topic_model=lsi,
                        total_topics=total_topics,
                        num_terms=15,
                        display_weights=False)


# In[ ]:


'''
Observations of the topics by MBTI type:
- Overall it appears the the same topics are extracted for each MBTI. This make sense since forum which the data is from
involves people from all types discusing topics. 
'''


# In[ ]:


#For each MBTI Type in the data, extract the top 20 topics 

total_topics = 20 #Number of topics

#Define the words to be removed from the lists
wordlist = ['url','infj','intj','infp','intp','enfj','entj','enfp','entp','isfj',
                'istj','isfp','istp','esfj','estj','esfp','estp','tapatalk']

print('-----------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------')
    
#Subset the data
subset = cleaned_mbti_token_cmtlvl
print(subset.shape)
print()

#Transform the data to list form
features = subset.iloc[:,5:].values.tolist()

#Remove common/superfuerlous words from the lists
feature_none = []
for x in features:
    y = list(filter(None.__ne__, x))
    z = [z for z in y if z not in wordlist]    
    feature_none.append(z)

#Create a dictionary of the words
dictionary = corpora.Dictionary(feature_none)
print(len(dictionary.token2id))

#Transform the document to a BOW
corpus = [dictionary.doc2bow(text, allow_update=True) for text in feature_none]
#print(corpus[:2])

#Transform to TFIDF
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

#Extract top topics using Latent Semantic Indexing
lsi = models.LsiModel(corpus_tfidf, 
                      id2word=dictionary, 
                      num_topics=total_topics)

#Print the top topics
print_topics_gensim(topic_model=lsi,
                        total_topics=total_topics,
                        num_terms=15,
                        display_weights=False)


# In[ ]:


'''
Observations of the topics:
- Welcome to the forum 
- Feelings
- Relationships/Friends
- Thanks to other uses comments

'''

