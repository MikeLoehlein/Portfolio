
# coding: utf-8

# In[ ]:


'''Introduction:
     Source Data from: https://www.kaggle.com/wendykan/lending-club-loan-data/data
    
     The goal of this script is to develop a model to estimate the probability of charge off given a loans origination
     characteristics. Only information available at origination will be used. Given that the portfolio is a mix of 
     current, paid off, charged off, and delinquent loans; all loans that are current will assume to stay current for the 
     life of the loan and all loans that are delinquent will be assumed to result in a charge-off.
'''


# In[ ]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Set Column Width to View all Columns in the Dataset
pd.set_option('display.max_columns', 500)


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
    print()
                        
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


#Load Dataset
dta = pd.read_csv("C:/Personal/Kaggle/lending-club-loan-data/loan.csv",dtype=object)


# In[ ]:


#Print  Column Names
print('Column Names are:')
print(dta.columns)

#Print Dataset Shape
print()
print("Shape is: " + str(dta.shape))

#View top five records
print()
print("Data frame head is: ")
print(dta.head())


# In[ ]:


#Calculate the percent of records with missing for each column
null_rate = dta.isnull().sum()/len(dta) 
print(null_rate)
print() 

#Create a list of columns to keep with no more than 25% missing variables
null_index = list(null_rate[null_rate < .75].index) 
print(null_index)
print()

#Remove  columns that are not needed or going to be used given that they are related to post origination performance
null_index.remove('id')
null_index.remove('member_id')
null_index.remove('emp_title')
null_index.remove('url')
null_index.remove('title')
null_index.remove('pymnt_plan')
null_index.remove('zip_code')
null_index.remove('initial_list_status')
null_index.remove('out_prncp') 
null_index.remove('out_prncp_inv')
null_index.remove('total_pymnt')
null_index.remove('total_rec_int')
null_index.remove('total_rec_prncp')
null_index.remove('total_pymnt_inv')
null_index.remove('total_rec_late_fee')
null_index.remove('recoveries')
null_index.remove('collection_recovery_fee')
null_index.remove('last_pymnt_d')
null_index.remove('last_pymnt_amnt')
null_index.remove('next_pymnt_d')
null_index.remove('last_credit_pull_d')
null_index.remove('mths_since_last_delinq')
null_index.remove('issue_d') 
null_index.remove('earliest_cr_line') 
null_index.remove('policy_code') 
null_index.remove('sub_grade') 
null_index.remove('application_type') 
null_index.remove('int_rate') 
null_index.remove('installment') 

#Subset the data to keep only selected columns
dta = dta[null_index]

#Print column names
print(dta.columns)
print()

#Print data head
print(dta.head())
print()


# In[ ]:


#Clean Data - Remove Rows with Missing Data

#For each column, print the percent of records with NAs
print(dta.isnull().sum()/len(dta))

#Remove Records with  NA's
print('shape origin - with NAs: ' + str(dta.shape))
dta.dropna(axis=0, how='any', inplace=True)
print('shape clean - without NAs: ' + str(dta.shape))


# In[ ]:


#Print Lists of Character Categories
print(dta.term.value_counts()) #Need to Create Dummy Variables For Modeling, '36 months' is reference class
print(dta.grade.value_counts(())) #Need to Create Dummy Variables For Modeling, 'A' is reference class
print(dta.emp_length.value_counts(())) #Recode NA and <1 year to 0; '0' is reference class
print(dta.home_ownership.value_counts(())) #Remove Other/None/Any and set as NA; 'mortgage' is reference class
print(dta.verification_status.value_counts(())) #"Source Verified" is reference class
print(dta.purpose.value_counts(())) #Recode to other if count is < 10,000. "debt_consolidation" is refernece class
print(dta.addr_state.value_counts(())) #"CA" is reference class

print(dta.loan_status.value_counts(())) #Dependent Variable, reclassify so it is only pay off 
        #and charge off and then encode so charge off = 1 and pay off = 0
      


# In[ ]:


#Record Values in Columns to Standardise and reduce the number of levels as needed
##### emp_length #####
mask = dta.emp_length.isin(['n/a'])
column_name = 'emp_length'
dta.loc[mask, column_name] = '< 1 year'
print(dta.emp_length.value_counts(()))
print()

##### home_ownership #####
mask = dta.home_ownership.isin(['OTHER','NONE','ANY'])
column_name = 'home_ownership'
dta.loc[mask, column_name] = 'OTHER'
print(dta.home_ownership.value_counts(()))
print()

##### purpose #####
counts = dta.purpose.value_counts()
counts = list(counts.loc[counts < 10000].index.values)
mask = dta.purpose.isin(counts)
column_name = 'purpose'
dta.loc[mask, column_name] = "other"
print(dta.purpose.value_counts(()))
print()

##### loan_status #####
chargeoff = ['Charged Off','Late (31-120 days)','In Grace Period','Late (16-30 days)','Default']
mask = dta.loan_status.isin(chargeoff)
column_name = 'loan_status'
dta.loc[mask, column_name] = "Default"
dta.loc[~mask, column_name] = "PayOff"

column_name = 'y'
dta.loc[mask, column_name] = 1
dta.loc[~mask, column_name] = 0
print(dta.y.value_counts(()))
print()

##### Grade #####
labeler = LabelEncoder()
labeler = labeler.fit(dta.grade)

y2 = labeler.transform(dta.grade)
y2 = pd.DataFrame(y2, columns=['y2'])
dta = dta.reset_index()
dta = pd.concat([dta, y2], axis=1)

print(dta.grade.value_counts(()))
print()


# In[ ]:


#Cast Columns to Correct Numeric Formats and Encodings; Strings can remain as objects
#Print Data Types
print(dta.dtypes)

#Print Head
print(dta.head(2))

#To Numeric
dta.loan_amnt = pd.to_numeric(dta.loan_amnt)
dta.funded_amnt = pd.to_numeric(dta.funded_amnt)
dta.funded_amnt_inv = pd.to_numeric(dta.funded_amnt_inv)
dta.annual_inc = pd.to_numeric(dta.annual_inc)
dta.dti = pd.to_numeric(dta.dti)
dta.delinq_2yrs = pd.to_numeric(dta.delinq_2yrs)
dta.inq_last_6mths = pd.to_numeric(dta.inq_last_6mths)
dta.open_acc = pd.to_numeric(dta.open_acc)
dta.pub_rec = pd.to_numeric(dta.pub_rec)
dta.revol_util = pd.to_numeric(dta.revol_util)
dta.revol_bal = pd.to_numeric(dta.revol_bal)
dta.total_acc = pd.to_numeric(dta.total_acc)
dta.collections_12_mths_ex_med = pd.to_numeric(dta.collections_12_mths_ex_med)
dta.acc_now_delinq = pd.to_numeric(dta.acc_now_delinq)
dta.tot_coll_amt = pd.to_numeric(dta.tot_coll_amt)
dta.tot_cur_bal = pd.to_numeric(dta.tot_cur_bal)
dta.total_rev_hi_lim = pd.to_numeric(dta.total_rev_hi_lim)


# In[ ]:


#Bivariate Exporation
#loan_status by ach potential explanatory variable
print('Mean acc_now_delinq by loan_status')
print(dta.groupby('loan_status').acc_now_delinq.mean())
print()

print('Mean annual_inc by loan_status')
print(dta.groupby('loan_status').annual_inc.mean())
print()

print('Mean collections_12_mths_ex_med by loan_status')
print(dta.groupby('loan_status').collections_12_mths_ex_med.mean())
print()

print('Mean delinq_2yrs by loan_status')
print(dta.groupby('loan_status').delinq_2yrs.mean())
print()

print('Mean dti by loan_status')
print(dta.groupby('loan_status').dti.mean())
print()

print('Mean funded_amnt by loan_status')
print(dta.groupby('loan_status').funded_amnt.mean())
print()

print('Mean funded_amnt_inv by loan_status')
print(dta.groupby('loan_status').funded_amnt_inv.mean())
print()

print('Mean loan_amnt by loan_status')
print(dta.groupby('loan_status').loan_amnt.mean())
print()

print('Mean inq_last_6mths by loan_status')
print(dta.groupby('loan_status').inq_last_6mths.mean())
print()

print('Mean open_acc by loan_status')
print(dta.groupby('loan_status').open_acc.mean())
print()

print('Mean pub_rec by loan_status')
print(dta.groupby('loan_status').pub_rec.mean())
print()

print('Mean revol_bal by loan_status')
print(dta.groupby('loan_status').revol_bal.mean())
print()

print('Mean revol_util by loan_status')
print(dta.groupby('loan_status').revol_util.mean())
print()

print('Mean tot_coll_amt by loan_status')
print(dta.groupby('loan_status').tot_coll_amt.mean())
print()

print('Mean tot_cur_bal by loan_status')
print(dta.groupby('loan_status').tot_cur_bal.mean())
print()

print('Mean total_acc by loan_status')
print(dta.groupby('loan_status').total_acc.mean())
print()

print('Mean total_rev_hi_lim by loan_status')
print(dta.groupby('loan_status').total_rev_hi_lim.mean())
print()

print('Term by loan_status')
print(dta.groupby(['term']).loan_status.value_counts(normalize=True))
print()

print('verification_status by loan_status')
print(dta.groupby(['verification_status']).loan_status.value_counts(normalize=True))
print()

print('addr_state by loan_status')
print(dta.groupby(['addr_state']).loan_status.value_counts(normalize=True))
print()

print('emp_length by loan_status')
print(dta.groupby(['emp_length']).loan_status.value_counts(normalize=True))
print()

print('home_ownership by loan_status')
print(dta.groupby(['home_ownership']).loan_status.value_counts(normalize=True))
print()

print('purpose by loan_status')
print(dta.groupby(['purpose']).loan_status.value_counts(normalize=True))
print()

print('grade by loan_status')
print(dta.groupby(['grade']).loan_status.value_counts(normalize=True))
print()



# In[ ]:


#Create Dummy Variables for Categorical Data
dta_dummied = pd.get_dummies(dta, 
                             drop_first=True, 
                             prefix=['term','emp_length','home_ownership','verification_status','purpose','addr_state'],
                             columns=['term','emp_length','home_ownership','verification_status','purpose','addr_state']                            )

#Remove Source Dependent Variables
dta_dummied.drop(['loan_status','grade'], axis=1, inplace=True)
print(dta_dummied.shape)
print()

#Remove any missing
print('Pre Cleaning Count: ' + str(dta_dummied.shape))
dta_dummied.dropna(axis=0, how='any', inplace=True)
print('Post Cleaning Count: ' + str(dta_dummied.shape))
print()

y = dta_dummied.y
y2 = dta_dummied.y2
print(y[:5])
print(y2[:5])

X = dta_dummied.drop(['y','y2'], axis=1)
print(X.columns)


# In[ ]:


#Split Data in to Train and Test Sets - Target Var = Default
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

#Print Shapes for Training data for Target Var Default
print('Shape of Default Data')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print()

#Split Data in to Train and Test Sets - Targe Var = Grade
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.10, random_state=42)

print('Shape of Grade Data')
print(X_train2.shape)
print(X_test2.shape)
print(y_train2.shape)
print(y_test2.shape)


# In[ ]:


#Train model for Default Estimate
print('Dependent Variable - Loand Status (Default = 1)')
lgr_default = LogisticRegression(penalty='l2', C=1, class_weight='balanced')

lgr_default_pred = train_predict_evaluate_model(classifier=lgr_default,
                                           train_features=X_train,
                                           train_labels=y_train,
                                           test_features=X_test,
                                           test_labels=y_test)

y_predicted = lgr_default.predict(X_test)

labels = list(set(y_test))
cm = metrics.confusion_matrix(y_test, y_predicted, labels)
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


print('Dependent Variable - Grade, predicted probability of default')
lgr_grade = LogisticRegression(penalty='l2', C=1, class_weight='balanced')

x_trainp = lgr_default.predict_proba(X_train2)
x_testp = lgr_default.predict_proba(X_test2)

lgr_grade_pred = train_predict_evaluate_model(classifier=lgr_grade,
                                           train_features=x_trainp,
                                           train_labels=y_train2,
                                           test_features=x_testp,
                                           test_labels=y_test2)

y_predicted = lgr_grade.predict(x_testp)
train = labeler.inverse_transform(y_test2)
test = labeler.inverse_transform(y_predicted)

labels = list(set(train))
cm = metrics.confusion_matrix(train, test, labels)
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


print('Dependent Variable - Grade, full origination data')
lgr_grade = LogisticRegression(penalty='l2', C=1, class_weight='balanced')

lgr_grade_pred = train_predict_evaluate_model(classifier=lgr_grade,
                                           train_features=X_train2,
                                           train_labels=y_train2,
                                           test_features=X_test2,
                                           test_labels=y_test2)

y_predicted = lgr_grade.predict(X_test2)
train = labeler.inverse_transform(y_test2)
test = labeler.inverse_transform(y_predicted)

import matplotlib.pyplot as plt

labels = list(set(train))
cm = metrics.confusion_matrix(train, test, labels)
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


print('Random Forest - Grade, full origination data')

rf = RandomForestClassifier(oob_score=True, n_jobs=2, n_estimators=250, 
                                        max_features='sqrt', criterion='gini', max_depth=None, 
                                        min_samples_split=10, min_samples_leaf=5, max_leaf_nodes=None)

rf_grade_pred = train_predict_evaluate_model(classifier=rf,
                                           train_features=X_train2,
                                           train_labels=y_train2,
                                           test_features=X_test2,
                                           test_labels=y_test2)

y_predicted = rf.predict(X_test2)
train = labeler.inverse_transform(y_test2)
test = labeler.inverse_transform(y_predicted)

labels = list(set(train))
cm = metrics.confusion_matrix(train, test, labels)
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


print('Random Forest - Default, full origination data')

rf = RandomForestClassifier(oob_score=True, n_jobs=2, n_estimators=250, 
                                        max_features='sqrt', criterion='gini', max_depth=None, 
                                        min_samples_split=10, min_samples_leaf=5, max_leaf_nodes=None)

rf_grade_pred = train_predict_evaluate_model(classifier=rf,
                                           train_features=X_train,
                                           train_labels=y_train,
                                           test_features=X_test,
                                           test_labels=y_test)

y_predicted = rf.predict(X_test2)

labels = list(set(y_test))
cm = metrics.confusion_matrix(y_test, y_predicted, labels)
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


'''
After removing all non-origination related information a logistic regression model is able to predict around 60% of paid in full vs 
charge-offs based only on origination data. While the overal rate is potentially acceptable, the model missed around 
25% of charge-offs and predicted a charge-off on a substantial amount of loans which are paid in full.

In terms of predicting the 'grade' of the borrower however, a logistic regression model was only able to predict 25% of the 'grades
of the loans. The initial model included interest rate and installement payment (whichare both a result of the grade) and
thus had a higher predictive power, however those variable were removed. Two models to predict the grade were evaluated: 1)
based on the full predcitor set and 2) based on the output from the charge-off model. They both performed about the same, with
a 1 in 4 accuracy.

The random forest model was able to predict grade better than a logistic regression model with about a 50% out of sample test. 
However the random forest model appears to over fit the data. For predicting default the random forest model failed to predict
any defaults and had 100% of the data predict to perform. Overal, I would not recommend a random forest model for the data as it
stands.
'''

