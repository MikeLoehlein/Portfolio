
# coding: utf-8

# In[ ]:


'2) Classification'

'''
This file estimates a classification model to predict the MBTI type of a person based on their comments.
The code does the following:
    - Loads libraries and functions
    - Imports the processed data saved in the "1) Data Prep" File
    - Splits the proccessed data in the label and feature datasets
    - Estimates the following models (as appropriate) on the loaded data
        - Support Vector Machine (BOW, TFIDF)
        - Logistic Regression (BOW, TFIDF, BOW SVD. BOW TFIDF)
        - Navie Bayes (BOW, TFIDF)
    - For the top models based on in-sample and out-of-sample metrics, prints the 16x16 confusion matrix
'''


# In[ ]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

#Import Selected Functions
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


#Load user defined functions

def get_metrics(true_labels, predicted_labels):
    '''Simple Function that prints the select functions
        Paramters:
        - true_labels: the 'true' or actual labels
        - predicted_labels: the predicted labels
    '''
    
    print('Accuracy:' + str(np.round(metrics.accuracy_score(true_labels, 
                                               predicted_labels),2)))
    
    print('Precision:' + str(np.round(metrics.precision_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),2)))
    
    print('Recall:' + str(np.round(metrics.recall_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),2)))
    
    print('F1 Score:' + str(np.round(metrics.f1_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),2)))
                        
def train_predict_evaluate_model(classifier, 
                                 train_features, train_labels, 
                                 test_features, test_labels):
    '''
    Function to estimate a classifier and print out the model fit metrics
    Paramters:
        - classifier: a sklearn algorithm object
        - train_features: the estimation features
        - train_labels: the estimation labels
        - test_features: the testing features
        - test_labels: the testing labels
    '''
    # build model    
    classifier.fit(train_features, train_labels)
    
    # predict using model
    predictions_insample = classifier.predict(train_features) 
    predictions = classifier.predict(test_features) 
    
    # evaluate model prediction performance   
    print("In of Sample Metrics")
    get_metrics(true_labels=train_labels, predicted_labels=predictions_insample)
    
    print("Out of Sample Metrics")
    get_metrics(true_labels=test_labels, predicted_labels=predictions)
    return predictions


# In[ ]:


#Load Data Sets - Non SVD
cleaned_bow_test = pd.read_pickle('cleaned_bow_test_userlvl.pkl')
cleaned_bow_train = pd.read_pickle('cleaned_bow_train_userlvl.pkl')

print("cleaned_bow_test shape: " + str(cleaned_bow_test.shape))
print("cleaned_bow_train shape: " + str(cleaned_bow_train.shape))

cleaned_tfidf_test = pd.read_pickle('cleaned_tfidf_test_userlvl.pkl')
cleaned_tfidf_train = pd.read_pickle('cleaned_tfidf_train_userlvl.pkl')

print("cleaned_tfidf_test shape: " + str(cleaned_tfidf_test.shape))
print("cleaned_tfidf_train shape: " + str(cleaned_tfidf_train.shape))

with open('LabelEncoder.pkl', 'rb') as pickle_file:
    lab_encoder = pickle.load(pickle_file)


# In[ ]:


#Load Data Sets - SVD
cleaned_bow_test_svd = pd.read_pickle('cleaned_bow_test_userlvl_svd.pkl')
cleaned_bow_train_svd = pd.read_pickle('cleaned_bow_train_userlvl_svd.pkl')

print("cleaned_bow_test - SVD shape: " + str(cleaned_bow_test_svd.shape))
print("cleaned_bow_train - SVD shape: " + str(cleaned_bow_train_svd.shape))

cleaned_tfidf_test_svd = pd.read_pickle('cleaned_tfidf_test_userlvl_svd.pkl')
cleaned_tfidf_train_svd = pd.read_pickle('cleaned_tfidf_train_userlvl_svd.pkl')

print("cleaned_tfidf_test - SVD shape: " + str(cleaned_tfidf_test_svd.shape))
print("cleaned_tfidf_train - SVD shape: " + str(cleaned_tfidf_train_svd.shape))


# In[ ]:


#BOW
#Training Data
print(cleaned_bow_train.head())

#BOW Training labels
bow_train_label = cleaned_bow_train['type_enc']
print('bow_train_label shape ' + str(bow_train_label.shape))
print(type(bow_train_label))
print(bow_train_label.head())
print()

#BOW Training features
bow_train_feature = cleaned_bow_train.iloc[:,2:]
print('bow_train_feature shape ' + str(bow_train_feature.shape))
print(type(bow_train_feature))
print(bow_train_feature.head())
print()

#Testing Data
print(cleaned_bow_test.head())

#BOW Testing labels
bow_test_label = cleaned_bow_test['type_enc']
print('bow_test_label shape ' + str(bow_test_label.shape))
print(type(bow_test_label))
print(bow_test_label.head())
print()

#BOW Testing Features
bow_test_feature = cleaned_bow_test.iloc[:,2:]
print('bow_test_feature shape ' + str(bow_test_feature.shape))
print(type(bow_test_feature))
print(bow_test_feature.head())
print()

############################################
#TF IDF

#Train Data
print(cleaned_tfidf_train.head())

#TFIDF Training labels
tfidf_train_label = cleaned_tfidf_train['type_enc']
print('tfidf_train_label shape ' + str(tfidf_train_label.shape))
print(type(tfidf_train_label))
print(tfidf_train_label.head())
print()

#TFIDF Training Features
tfidf_train_feature = cleaned_tfidf_train.iloc[:,2:]
print('tfidf_train_feature shape ' + str(tfidf_train_feature.shape))
print(type(tfidf_train_feature))
print(tfidf_train_feature.head())
print()

#Test Data
print(cleaned_tfidf_test.head())

#TFIDF Testing labels
tfidf_test_label = cleaned_tfidf_test['type_enc']
print('tfidf_test_label shape ' + str(tfidf_test_label.shape))
print(type(tfidf_test_label))
print(tfidf_test_label.head())
print()

#TFIDF Testing Features
tfidf_test_feature = cleaned_tfidf_test.iloc[:,2:]
print('tfidf_test_feature shape ' + str(tfidf_test_feature.shape))
print(type(tfidf_test_feature))
print(tfidf_test_feature.head())
print()


# In[ ]:


#SVD Data
#BOW
#Training Data
print(cleaned_bow_train_svd.head())

#BOW – SVD Training labels
bow_train_label_svd = cleaned_bow_train_svd['type_enc']
print('bow_train_label shape ' + str(bow_train_label_svd.shape))
print(type(bow_train_label_svd))
print(bow_train_label_svd.head())
print()

#BOW – SVD Training features
bow_train_feature_svd = cleaned_bow_train_svd.iloc[:,2:]
print('bow_train_feature shape ' + str(bow_train_feature_svd.shape))
print(type(bow_train_feature_svd))
print(bow_train_feature_svd.head())
print()

#Testing Data
print(cleaned_bow_test_svd.head())

#BOW – SVD Testing labels
bow_test_label_svd = cleaned_bow_test_svd['type_enc']
print('bow_test_label shape ' + str(bow_test_label_svd.shape))
print(type(bow_test_label_svd))
print(bow_test_label_svd.head())
print()

#BOW – SVD Testing Features
bow_test_feature_svd = cleaned_bow_test_svd.iloc[:,2:]
print('bow_test_feature shape ' + str(bow_test_feature_svd.shape))
print(type(bow_test_feature_svd))
print(bow_test_feature_svd.head())
print()

############################################
#TF IDF

#Train Data
print(cleaned_tfidf_train_svd.head())

#TFIDF – SVD Training labels
tfidf_train_label_svd = cleaned_tfidf_train_svd['type_enc']
print('tfidf_train_label shape ' + str(tfidf_train_label_svd.shape))
print(type(tfidf_train_label_svd))
print(tfidf_train_label_svd.head())
print()

#TFIDF – SVD Training Features
tfidf_train_feature_svd = cleaned_tfidf_train_svd.iloc[:,2:]
print('tfidf_train_feature shape ' + str(tfidf_train_feature_svd.shape))
print(type(tfidf_train_feature_svd))
print(tfidf_train_feature_svd.head())
print()

#Test Data
print(cleaned_tfidf_test_svd.head())

#TFIDF – SVD Testing labels
tfidf_test_label_svd = cleaned_tfidf_test_svd['type_enc']
print('tfidf_test_label shape ' + str(tfidf_test_label_svd.shape))
print(type(tfidf_test_label_svd))
print(tfidf_test_label_svd.head())
print()

#TFIDF – SVD Testing Features
tfidf_test_feature_svd = cleaned_tfidf_test_svd.iloc[:,2:]
print('tfidf_test_feature shape ' + str(tfidf_test_feature_svd.shape))
print(type(tfidf_test_feature_svd))
print(tfidf_test_feature_svd.head())
print()


# In[ ]:


#Initialize Classification Models
mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge', max_iter=250, class_weight='balanced')
lgr = LogisticRegression(penalty='l2', C=.1, class_weight='balanced')

# Logistic Regression with bag of words features - SVD
print()
print("Logistic Regression for BoW - SVD")
lgr_bow_predictions_svd = train_predict_evaluate_model(classifier=lgr,
                                           train_features=bow_train_feature_svd,
                                           train_labels=bow_train_label_svd,
                                           test_features=bow_test_feature_svd,
                                           test_labels=bow_test_label_svd)
                                           
# Logistic Regression with tfidf features - SVD
print()
print("Logistic Regression for TF-IDF - SVD")
lgr_tfidf_predictions_svd = train_predict_evaluate_model(classifier=lgr,
                                           train_features=tfidf_train_feature_svd,
                                           train_labels=tfidf_train_label_svd,
                                           test_features=tfidf_test_feature_svd,
                                           test_labels=tfidf_test_label_svd)

# Logistic Regression with bag of words features
print()
print("Logistic Regression for BoW")
lgr_bow_predictions = train_predict_evaluate_model(classifier=lgr,
                                           train_features=bow_train_feature,
                                           train_labels=bow_train_label,
                                           test_features=bow_test_feature,
                                           test_labels=bow_test_label)
                                           
# Logistic Regression with tfidf features
print()
print("Logistic Regression for TF-IDF")
lgr_tfidf_predictions = train_predict_evaluate_model(classifier=lgr,
                                           train_features=tfidf_train_feature,
                                           train_labels=tfidf_train_label,
                                           test_features=tfidf_test_feature,
                                           test_labels=tfidf_test_label)

# Multinomial Naive Bayes with bag of words features
print()
print("MultinomialNB for BoW")
mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                           train_features=bow_train_feature,
                                           train_labels=bow_train_label,
                                           test_features=bow_test_feature,
                                           test_labels=bow_test_label)

# Support Vector Machine with bag of words features
print()
print("SVM for BoW")
svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=bow_train_feature,
                                           train_labels=bow_train_label,
                                           test_features=bow_test_feature,
                                           test_labels=bow_test_label)
                                           
# Multinomial Naive Bayes with tfidf features
print()
print("MultinomialNB for TF-IDF")
mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                           train_features=tfidf_train_feature,
                                           train_labels=tfidf_train_label,
                                           test_features=tfidf_test_feature,
                                           test_labels=tfidf_test_label)

# Support Vector Machine with tfidf features
print()
print("SVM for TF-IDF")
svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=tfidf_train_feature,
                                           train_labels=tfidf_train_label,
                                           test_features=tfidf_test_feature,
                                           test_labels=tfidf_test_label)


# In[ ]:


print("LGR TF-IDF - SVD")
print()

tfidf_test_label2 = lab_encoder.inverse_transform(tfidf_test_label)
mnb_tfidf_predictions2 = lab_encoder.inverse_transform(lgr_tfidf_predictions_svd)

labels = list(set(tfidf_test_label2))
cm = metrics.confusion_matrix(tfidf_test_label2, mnb_tfidf_predictions2, labels)
cm2 = pd.DataFrame(cm, index=labels, columns=labels)
print(cm2)
cm = cm / cm.astype(np.float).sum(axis=1)

fig = plt.figure(figsize=(25,10))
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:


#Deep Dive Into Selected Model
# Logistic Regression with tfidf features - SVD
lgr = LogisticRegression(penalty='l2', C=.1, class_weight='balanced')

# build model    
lgr.fit(tfidf_train_feature_svd, tfidf_train_label_svd)
    
# predict using model
predictions_insample = lgr.predict(tfidf_train_feature_svd) 
predictions = lgr.predict(tfidf_test_feature_svd) 
    
# evaluate model prediction performance   
print("In of Sample Metrics")
get_metrics(true_labels=tfidf_train_label_svd, predicted_labels=predictions_insample)
    
print("Out of Sample Metrics")
get_metrics(true_labels=tfidf_test_label_svd, predicted_labels=predictions)


# In[ ]:


#Deep Dive Into Selected Model
# Logistic Regression with tfidf features - SVD
lgr = LogisticRegression(penalty='l2', C=.1, class_weight='balanced')

# build model    
lgr.fit(tfidf_train_feature_svd, tfidf_train_label_svd)
    
# predict using model
predictions_insample = lgr.predict(tfidf_train_feature_svd) 
predictions = lgr.predict(tfidf_test_feature_svd) 
    
# evaluate model prediction performance   
print("In of Sample Metrics")
get_metrics(true_labels=tfidf_train_label_svd, predicted_labels=predictions_insample)

print()
print("Out of Sample Metrics")
get_metrics(true_labels=tfidf_test_label_svd, predicted_labels=predictions)

#Percent when correct label is in top three predicted labels ranked by predicted probability
predictions_proba = lgr.predict_proba(tfidf_test_feature_svd) 

names = list(set(tfidf_test_label_svd))
dict_list = []
for p in predictions_proba:
    q = dict(zip(names,p))
    t = dict(sorted(q.items(), key=lambda x:-x[1])[:3])
    dict_list.append(t)
    

results = []
for p in range(len(dict_list)-1):
    if tfidf_test_label_svd[p] in dict_list[p].keys():
        u = 1
    else:
        u = 0
    results.append(u)
    
print()    
print("{0:.1f} Percent of Correct Prediction when Correct Label is in the top 3 Predicted Labels ".format(np.mean(results)*100))f

