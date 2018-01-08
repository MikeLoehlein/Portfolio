
# coding: utf-8

# In[ ]:


'1) Data Prep'

'''
This file prepares the MBTI source data. It does the following:
    - Imports the MBTI data sourced from: https://www.kaggle.com/datasnaek/mbti-type
    - Performs a basic spell check on the data
    - Normalizes the data
        - Normalizes without tokenization
        - Normalizes with tokenization
    - Performs SVD on the data to reduce the features
    - Saves the cleaned data a set of pickle files for modeling
'''


# In[ ]:


####Import libraries and modules
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
import collections
import pickle

#Import Functions from Libraries
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.decomposition import TruncatedSVD
from datetime import datetime
from sklearn.model_selection import train_test_split

#Import user defined functions
from feature_extractors import *
from normalization import *
from spelling_corrector import *
from contractions import *

def bow_extractor(corpus, ngram_range=(1,1)):
    
    vectorizer = CountVectorizer(max_df=.85, min_df=.01, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


# In[ ]:


#Initalize Stop Words and Lemmatizer
stopword_list = nltk.corpus.stopwords.words('english')
wnl = WordNetLemmatizer() 


# In[ ]:


####Load the data
mbti = pd.read_csv("C:/Personal/Kaggle/mbti-myers-briggs-personality-type-dataset/mbti_1.csv") 
#Source: https://www.kaggle.com/datasnaek/mbti-type

#Print top five rows
print(mbti.head(5))


# In[ ]:


#Plot MBTI Type Distrbution
plt.figure(figsize=(40,20))
plt.xticks(fontsize=24, rotation=0)
plt.yticks(fontsize=24, rotation=0)
sns.countplot(data=mbti, x='type')
plt.show()


# In[ ]:


####Reshape the Data into One row per comment per user

#Split the comments into a list of lists
dta = [p.split('|||') for p in mbti.posts.values]

#Cast as a dataframe
df = pd.DataFrame(dta)

#Set miissing as NAN
df[df==""] = np.nan
df[df.isnull()] = np.nan

#Remove extra columns
df = df.dropna(axis=1,how='all')

#Create column labels
b = []
for i in list(range(1, df.shape[1]+1)):
    b.append('s' + str(i))

#stack the columns
df.columns = b
df = df.stack()
df = df.reset_index()

#Merge the processed posts with the original MBTI type
df = df.rename(columns={'level_0': 'index'})
mbti2 = mbti.reset_index().drop('posts',axis=1)
df = pd.merge(df, mbti2, on='index')
df = df.rename(columns={'index': 'user','level_1': 'commentnum', 0:'comment'})

print()
print("Top Five Rows")
print(df.head())

print()
print("Shape of Dataframe")
print(df.shape)

print()
#Plot Number of comments Per User
mean_comment = df.groupby('user').agg('count')['commentnum'].mean()
print("Average number of comments per user " + str(round(mean_comment)))

mean_comment = df.groupby('user').agg('count')['commentnum'].min()
print("Min number of comments per user " + str(mean_comment))

mean_comment = df.groupby('user').agg('count')['commentnum'].max()
print("Max number of comments per user " + str(mean_comment))


# In[ ]:


#Convert MBIT to Class
unique_type_list = df.type.unique()
lab_encoder = LabelEncoder().fit(unique_type_list)

with open("LabelEncoder.pkl", "wb") as f:
    pickle.dump(lab_encoder, f, pickle.HIGHEST_PROTOCOL)

df['type_enc'] = lab_encoder.transform(df['type'])

#Replace Links
df['comment'] = df.comment.str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','url')

#Split comments based in pipes, replace with white space
df['comment'] = df.comment.str.replace('\|+',' ')

#Keep Only ASCII
df['comment'] = df.comment.str.replace('[^\x00-\x7F]+','')

#Remove Selected Puncation
df['comment'] = df.comment.str.replace('[,.:!@#$%&*()_+?><]+',' ')

#Remove Words Longer than 20 characters
df['comment'] = df.comment.str.replace('[a-zA-Z0-9_]{20,}',' ')                                       
                                       
#Remove Extra White Space
df['comment'] = df.comment.str.replace('[\s]+',' ')                                       
                                                                              
print(df.head())


# In[ ]:


#Correct Spelling
#df = df[0:200] #Sample For Testing

#Correct Spelling
comments = df['comment'].tolist()
print(len(comments))
print(type(comments))
print()

dta = []
i = 0
for x in comments:
    if (i % 100 == 0):
        print(str(datetime.datetime.now()))
        print("Record " + str(i) + " of " + str(len(comments)))
    x = ' '.join([correct_text_generic(y) for y in x.split(' ')])
    dta.append(x)
    i += 1
                 
print(dta[0:2])
print()


# In[ ]:


###Normalize Data
#No Tokenization
dta_notoken = normalize_corpus(dta, tokenize=False, contraction=CONTRACTION_MAP)
print(dta_notoken[0:2])
print()

clean = pd.DataFrame({'clean_comment':dta_notoken})
cleaned = pd.concat([df, clean], axis=1)
print(cleaned[0:2])
print()

cleaned.to_pickle("cleaned_mbti_cmtlvl.pkl")

####################################################################3
#Tokenization
dta_token = normalize_corpus(dta, tokenize=True, contraction=CONTRACTION_MAP)
print(dta_token[0:2])
print()

clean = pd.DataFrame(dta_token)
cleaned = pd.concat([df, clean], axis=1)
print(cleaned[0:2])
print()

cleaned.to_pickle("cleaned_mbti_token_cmtlvl.pkl")


# In[ ]:


#Train Test Split
X = dta_notoken
y = df['type_enc']
print(len(X))
print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)

print(len(X_train))
print(X_train[:2])
print()

print(len(X_test))
print(X_test[:2])
print()

print(len(y_train))
print(y_train[:2])
print()

print(len(y_test))
print(y_test[:2])
print()


# In[ ]:


#Feature Extraction
bow_vectorizor, bow_features = bow_extractor(X_train, ngram_range=(1,1))
bow_features = bow_features.todense()
print("bow_features shape: " + str(bow_features.shape))

tfidf_vectorizor, tfidf_features = tfidf_transformer(bow_features)
tfidf_features = tfidf_features.todense()
print("tfidf_features shape: " + str(tfidf_features.shape))

bow_features_test = bow_vectorizor.transform(X_test)
bow_features_test = bow_features_test.todense()
print("bow_features_test shape: " + str(bow_features_test.shape))

tfidf_features_test = tfidf_vectorizor.transform(bow_features_test)
tfidf_features_test = tfidf_features_test.todense()
print("tfidf_features_test shape: " + str(tfidf_features_test.shape))


# In[ ]:


#Save Processed Data to Disk
clean_bow = pd.DataFrame(bow_features)
cleaned_bow = pd.concat([y_train.reset_index(), clean_bow], axis=1)
cleaned_bow.to_pickle("cleaned_bow_train_cmtlvl.pkl")

clean_tfidf = pd.DataFrame(tfidf_features)
cleaned_tfidf = pd.concat([y_train.reset_index(), clean_tfidf], axis=1)
cleaned_tfidf.to_pickle("cleaned_tfidf_train_cmtlvl.pkl")

clean_bow = pd.DataFrame(bow_features_test)
cleaned_bow = pd.concat([y_test.reset_index(), clean_bow], axis=1)
cleaned_bow.to_pickle("cleaned_bow_test_cmtlvl.pkl")

clean_tfidf = pd.DataFrame(tfidf_features_test)
cleaned_tfidf = pd.concat([y_test.reset_index(), clean_tfidf], axis=1)
cleaned_tfidf.to_pickle("cleaned_tfidf_test_cmtlvl.pkl")


# In[ ]:


#SVD

#BOW SVD
svd = TruncatedSVD(n_components=150)
svd.fit(bow_features)
bow_features_svd = svd.transform(bow_features)
bow_features_svd = pd.DataFrame(bow_features_svd)
bow_features_svd = pd.concat([y_train.reset_index(), bow_features_svd], axis=1)
bow_features_svd.to_pickle("cleaned_bow_train_cmtlvl_svd.pkl")

bow_features_test_svd = svd.transform(bow_features_test)
bow_features_test_svd = pd.DataFrame(bow_features_test_svd)
bow_features_test_svd = pd.concat([y_test.reset_index(), bow_features_test_svd], axis=1)
bow_features_test_svd.to_pickle("cleaned_bow_test_cmtlvl_svd.pkl")

#TFIDF SVD
svd = TruncatedSVD(n_components=150)
svd.fit(tfidf_features)
tfidf_features_svd = svd.transform(tfidf_features)
tfidf_features_svd = pd.DataFrame(tfidf_features_svd)
tfidf_features_svd = pd.concat([y_train.reset_index(), tfidf_features_svd], axis=1)
tfidf_features_svd.to_pickle("cleaned_tfidf_train_cmtlvl_svd.pkl")

tfidf_features_test_svd = svd.transform(tfidf_features_test)
tfidf_features_test_svd = pd.DataFrame(tfidf_features_test_svd)
tfidf_features_test_svd = pd.concat([y_test.reset_index(), tfidf_features_test_svd], axis=1)
tfidf_features_test_svd.to_pickle("cleaned_tfidf_test_cmtlvl_svd.pkl")

