
# coding: utf-8

# In[ ]:


'''4) Sentiment Analysis'''

'''
This file takes the comment level cleaned data from 1) Data Prep and preforms unsupervised sentiment analysis on it. 
The code uses three seperate lexicon based scoring methods and compares the results across the users. 

The code does the following:
 - Import libraries
 - Defines functions
 - Subsets and formats the data. This process uses the cleaned and processed data, not the raw comments.
 - Scores the data
 - Compares the scored sentiment accoring to 1) overal and 2) MBTI type.
'''


# In[ ]:


#Import Libaries and Modules
import pandas as pd
import numpy as np
import pickle
import nltk

from afinn import Afinn
afn = Afinn(emoticons=False) 

from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from normalization import *


# In[ ]:


#Define Functions
def analyze_sentiment_sentiwordnet_lexicon(review):
    # tokenize and POS tag text tokens
    text_tokens = nltk.word_tokenize(review)
    tagged_text = nltk.pos_tag(text_tokens)
    pos_score = neg_score = token_count = obj_score = 0
    # get wordnet synsets based on POS tags
    # get sentiment scores if synsets are found
    for word, tag in tagged_text:
        ss_set = None
        if 'NN' in tag and swn.senti_synsets(word, 'n'):
            ss_set = list(swn.senti_synsets(word, 'n'))
        elif 'VB' in tag and swn.senti_synsets(word, 'v'):
            ss_set = list(swn.senti_synsets(word, 'v'))
        elif 'JJ' in tag and swn.senti_synsets(word, 'a'):
            ss_set = list(swn.senti_synsets(word, 'a'))
        elif 'RB' in tag and swn.senti_synsets(word, 'r'):
            ss_set = list(swn.senti_synsets(word, 'r'))
        # if senti-synset is found        
        if ss_set:
            ss_set = ss_set[0]
            # add scores for all found synsets
            pos_score += ss_set.pos_score()
            neg_score += ss_set.neg_score()
            obj_score += ss_set.obj_score()
            token_count += 1
    
    # aggregate final scores
    try:
        final_score = pos_score - neg_score
        norm_final_score = round(float(final_score) / token_count, 2)
        final_sentiment = 'positive' if norm_final_score >= 0 else 'negative'
    except:
        final_sentiment = 'Not Evaluated'
        
    return final_sentiment

def analyze_sentiment_vader_lexicon(review, 
                                    threshold=0.1):
    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)
    # get aggregate scores and final sentiment
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold                                   else 'negative'
   
    return final_sentiment

def freq(lst):
    d = {}
    for i in lst:
        if d.get(i):
            d[i] += 1
        else:
            d[i] = 1
    return d


# In[ ]:


#Load Data
cleaned_mbti_userlvl = pd.read_pickle('cleaned_mbti_cmtlvl.pkl')
comments = np.array(cleaned_mbti_userlvl['clean_comment']) #Cast to an array
print(cleaned_mbti_userlvl.head(2))
print()


# In[ ]:


#AFINN Sentiment Analysis Scoring
afinn_score = [afn.score(review) for review in comments]

afinn_score_final = []

for x in afinn_score:
    if x > 0:
        hold = 'positive'
    else:
        hold = "negative"
    afinn_score_final.append(hold)


# In[ ]:


#Sentiment Analysis using Vader Lexicon
vader_predictions = [analyze_sentiment_vader_lexicon(review, threshold=0.1) for review in comments] 


# In[ ]:


#Sentiment Analysis using SentiWord Lexicon
sentiwordnet_predictions = [analyze_sentiment_sentiwordnet_lexicon(review) for review in comments]


# In[ ]:


#Print Scores for first five
print('afinn_score ' + str(len(afinn_score_final)))
print(afinn_score_final[:5])

print('vader_predictions ' + str(len(vader_predictions)))
print(vader_predictions[:5])

print('sentiwordnet_predictions ' + str(len(sentiwordnet_predictions)))
print(sentiwordnet_predictions[:5])


# In[ ]:


#Print Classification Summary
print('AFINN Classification ' + str(freq(afinn_score_final)))
print('VADER Classification ' + str(freq(vader_predictions)))
print('SentiWordNet Classification ' + str(freq(sentiwordnet_predictions)))


# In[ ]:


type = cleaned_mbti_userlvl['type']
AFINN = pd.DataFrame(afinn_score_final, columns=['afinn'])
VADER = pd.DataFrame(vader_predictions, columns=['vader'])
SentiWord = pd.DataFrame(sentiwordnet_predictions, columns=['senti'])

Combined = pd.concat([type, AFINN, VADER, SentiWord], axis=1)
Combined['concur'] = np.where((Combined['afinn'] >= Combined['vader']) & (Combined['afinn'] <= Combined['senti'])
                     , 'concurrent', 'not')


print(Combined.head())
print()

counts = Combined.concur.value_counts(normalize=True)
print(counts)
print()

counts = Combined.groupby('afinn').concur.value_counts(normalize=True)
print(counts)
print()

counts = Combined.groupby('type').concur.value_counts(normalize=True)
print(counts)
print()

counts = Combined.groupby(['type','afinn']).concur.value_counts(normalize=True)
print(counts)
print()


# In[ ]:


'''
Comments:
    - Using the comment level data inplies that overal the comments are at least 50% positive. Depending on the lexicon used 
    the positive vs negatiev ratio goes from 50% postive to about 75% positive. 
    
    - Across MBTI type, the percent of comment which the three lexicons scored as conccurrent (so all three as positive or 
    all three as negative is farily stable with around 20% non currence by MBTI and scoring (positive vs negative)).
    
    - The comments which are concurrent (so all positive or all negative) are likely properly scored. However the non current 
    comments should be mannually reviewed and classified.

