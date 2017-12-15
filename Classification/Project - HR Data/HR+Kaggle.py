
# coding: utf-8

# In[ ]:


#Introdution:
# This Python code file estimates a seires of classifiation models with the goal of predicting whether an 
# employee will leave a company or not. The data is sourced from Kaggle: https://www.kaggle.com/ludobenistant/hr-analytics.
# The code below has several sections: 
# - Load libraries and user defined functions
# - Load the data and perform exploratory data analysis using univariate and bivariate charts and tables
# - Prepare the data for modeling
# - Estimate classifiers:
#   - Logistic Regression
#   - Support Vecotor Machine
#   - Random Forest Decision Tree
#   - Naive Bayes
#   - Ensamble of Logistic Regression, Random Forest Decision Tree, and Naive Bayes
# For each classifier, there are two sections: 1) hyper paramter grid search, and 2) model estimation with the hyper parameters


# In[ ]:


#Load libraries and define functions

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

#User Defined Functions
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """Print a confusion matrix with user provided labels """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
        
def classification_output_report(X_train, y_train, X_test, y_test, y_pred, y_pred_prob, train_accuracy, test_accuracy):
    '''For a given set of classification model inputs and outputs, print a standard set of outputs
       including model accuracy, confusion matrix, classification report, and ROC Curve/AUC. 
       
       Parameters:
           - X_train: The model training data
           - y_train: The model training labels
           - X_test: The model testing data
           - y_test: The model testing labels
           - y_pred: The predicted labels, from model.predict()
           - y_pred_prob: The predicted label probabilities, from model.predict_log_proba()
           - train_accuracy: The accuracy of the model on the training data, from model.score(X_train, y_train)
           - test_accuracy: The accuracy of the model on the testing data, from model.score(X_test, y_test)
           
       Note: If a model does not by default estimate a probability then y_pred_prob should be set to None'''
    
    #Print In and Out of Sample Accuracy
    diff_accuracy = train_accuracy - test_accuracy

    print('Accuracy of classifier on train set: {:.2f}'.format(train_accuracy))
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(test_accuracy))
    print('Difference in Accuracy - Train Minus Test: {:.8f}'.format(diff_accuracy))

    #Confusion Matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    print()
    print("Confusion Matrix")
    print_cm(cf_matrix, ["Stayed", "Left"])

    #Classification Report
    print()
    print("Classification Report")
    print(classification_report(y_test, y_pred))

    #ROC Curve + AUC
    if y_pred_prob is None:
        print("")
    else:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
        roc_auc = roc_auc_score(y_test, y_pred)
        plt.figure()
        plt.plot(fpr, tpr, label='AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()        


def grid_search(scores, model, param, name):
    '''Perform a grid search over a set of hyper paramters for a model
    
       Parameters:
           - scores: The list of metrics to be evaluated for, Options: accuracy, recall, precision, f1 
           - model: The function name of the model
           - param: A dictionary of parameters to search over
           - name: A string with the name of the model to use in printing the report
    '''
    for score in scores:
        print('Evaluating the ' + name + 'for optimal hyperparamters')
        print("# Tuning hyper-parameters for %s" % score)
        print()

        #Perform grid search using cross validation
        clf = GridSearchCV(model(), param, cv=10, scoring='%s' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        
#EDA Functions
def numeric_eda_scripts(table, column):
    '''
    Numeric EDA script which plots a historgraph of the data and the mean, median, and 25th/75th percentiles.
    
    Parameters
        - table: The table name
        - column: The column name, formatted as a string
    '''
    print("Number of NA values: " + str(table[column].isnull().sum()))
    
    plt.hist(table[column], bins=20, rwidth=.9)
    plt.title("Histograph of " + column.title() + " - Total Records = " + str(len(table)))
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()

    print("Average " + column.title() + ": " + str(np.round(np.mean(table[column]),2)))
    print("Median " + column.title() + ": " + str(np.median(table[column])))
    print("25 Percentile " + column.title() + ": " + str(np.percentile(table[column], 25)))
    print("75 Percentile " + column.title() + ": " + str(np.percentile(table[column], 75))) 

def cat_eda_scripts(table, column): 
    '''
    Numeric EDA script which print the value counts and plots a historgraph of the value counts of the specified column.
    
    Parameters
        - table: The table name
        - column: The column name, formatted as a string
    '''
    data = table
    data = data[column].apply(str)

    data.value_counts().plot(kind='bar')
    plt.title("Histograph of " + column + " - Total Records = " + str(len(data)))
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()

    print(data.value_counts())
    
def numeric_by_cat_eda(table, col1, col2):
    '''
    Numeric EDA script which plots a grouped boxplot of the numeric data (col1) by the categorical data (col2).
    
    Parameters
        - table: The table name
        - col1: The numeric column name, formatted as a string
        - col1: The cateogrical column name, formatted as a string
    '''
    data = table
    data[col2] = data[col2].apply(str)

    plt.title("Group Box Plot of " + col1 + " by " + col2)
    p = sns.boxplot(y=col1, x=col2, data=data)
    plt.show()
    
    
def cat_by_cat_eda_scripts(table, col1, col2):
    '''
    Categorical EDA script which plots a grouped bar chart of the data (col1) by the data (col2). Two plots are 
    created, one of the value counts and one of the percentages within the by group.
    
    Parameters
        - table: The table name
        - col1: The 1st categorical column name, formatted as a string
        - col1: The 2nd cateogrical column name, formatted as a string
    '''
    
    #Plot value counts by group
    plt.axes([0.05,0.05,0.425,0.9])
    plt.title('Counts by Group - ' + col2)
    sns.countplot(x=col1, orient='v', hue=col2, data=table, palette="Greens_d")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    #Plot percentages by group
    plt.axes([.8,0.05,0.425,0.9])
    rates = (table.groupby([col2])[col1]
                         .value_counts(normalize=True)
                         .rename('percentage')
                         .mul(100)
                         .reset_index()
                         .sort_values(col1))

    plt.title('Percentages by Group - ' + col2)
    p = sns.barplot(x=col1, y="percentage", hue=col2, data=rates, palette="Greens_d")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
        


# In[ ]:


#Load the data
hr_data = pd.read_csv("C:/Personal/Kaggle/human-resources-analytics/Data/HR_comma_sep.csv") 
#Source: https://www.kaggle.com/ludobenistant/hr-analytics

print("Number of Rows: " + str(len(hr_data))) #Check Row Count

print()
print("First Five Rows:")
print(hr_data.iloc[0:5]) #Check First 5 Rows

print()
print("Final 5 Rows:")
print(hr_data.iloc[-5:]) #Check Final 5 Rows

print()
print("Column Data Types")
print(hr_data.dtypes) #Print the colum types


# In[ ]:


#EDA for Satisfaction Level
numeric_eda_scripts(hr_data, "satisfaction_level")

print('''
      Analyst Comments/Observations:
      Based on the metadata provided, satisfaction_level is defined 
      as: "Level of satisfaction (0-1)".
      
      There are no missing/NA values within this column. In terms of 
      data quality, all of the data points are between 0 and 1 
      which gives reasonable assurance that the data has no outliers.

      Satisfaction level appears to follow a bimodal distribution with a 
      spike in the number employees having very low levels of satisfication (<0.2) and 
      the majority of employees having higher levels of satisfacation (>0.4 
      or >0.5). The 25th percentile and 75th percentile are about the same 
      distance away from the median (distance of 0.18 to 0.20) which implies 
      the body of the distribution is symetrical.
      ''')


# In[ ]:


#EDA for Last Evaluation
numeric_eda_scripts(hr_data, "last_evaluation")

print('''
      Analyst Comments/Observations:
      Based on the metadata provided, satisfaction_level is defined 
      as: "Time since last performance evaluation (in Years)".
      
      There are no missing/NA values within this column. In terms of 
      data quality, all of the data points are between 0 and 1. This means
      that all employees have had an evaluation in the past year. This implies
      that there are not outliers in the data.

      The data is multi modal with spikes in the data at 1, about .5 and at about .85.
      The 25th percentile and 75th percentile are about the same 
      distance away from the median (distance of 0.16) which implies 
      the body of the distribution is symetrical. 
      ''')


# In[ ]:


#EDA for Average Monthly Hours
numeric_eda_scripts(hr_data, "average_montly_hours")

print('''
      Analyst Comments/Observations:
      Based on the metadata provided, average_montly_hours is defined 
      as: "Average monthly hours at workplace".
      
      There are no missing/NA values within this column. The prior assumption 
      that a standard work week is 40 hours per week means that there is on 
      average 160 or so hours in a month. If a worker is doing more than 
      160 than they are working overtime. The data shows that no employee is 
      below about 100 hours per month. The mode is at about 160. However the data is 
      shows a lot of values above 160. The max values appear at about 300 hours, which 
      imply a 75 hour work week, which is plausible but very high.

      The 25th percentile and 75th percentile are about the same 
      distance away from the median (distance of about 45) which implies 
      the body of the distribution is symetrical. 
      ''')


# In[ ]:


#EDA for Promotion in the Last 5 Years
cat_eda_scripts(hr_data, "promotion_last_5years")

print('''
      Analyst Comments/Observations:
      Based on the metadata provided, promotion_last_5years is defined 
      as: "Whether the employee was promoted in the last five years".
      
      Most employee's did not get a promotion.       
      ''')


# In[ ]:


#EDA for Left
cat_eda_scripts(hr_data, "left")

print('''
      Analyst Comments/Observations:
      Based on the metadata provided, left is defined 
      as: "Whether the employee left the workplace or not (1 or 0)".
      
      Most employee's did not leave though about 23% did.
      ''')


# In[ ]:


#EDA for Number of Projects
numeric_eda_scripts(hr_data, "number_project")

print('''
      Analyst Comments/Observations:
      Based on the metadata provided, number_project is defined 
      as: "Number of projects completed while at work".
    
      There are no missing values in this column. Range of values
      is between 2 and 7.
      
      ''')


# In[ ]:


#EDA for Time Spent with the Company
numeric_eda_scripts(hr_data, "time_spend_company")

print('''
      Analyst Comments/Observations:
      Based on the metadata provided, time_spend_company is defined 
      as: "Number of years spent in the company".
      
      From the histogram it looks like the employees at the company 
      all have been there at least 1 year though most are either 2 or 3 year tenured. 
      This could imply that after 3 years that there is a natural churn at the company.
      ''')


# In[ ]:


#EDA for Work Accident
cat_eda_scripts(hr_data, "Work_accident")

print('''
      Analyst Comments/Observations:
      Based on the metadata provided, Work_accident is defined 
      as: "Whether the employee had a workplace accident".
       
      2000 (15%) of employees had a work accident. This seems like a 
      high rate.
      ''')


# In[ ]:


#EDA for Sales
cat_eda_scripts(hr_data, "sales")

print('''
      Analyst Comments/Observations:
      Based on the metadata provided, sales is defined 
      as: "Department in which they work for".
       
      The largest number of employees are in sales and techincal departments.
      ''')


# In[ ]:


#EDA for Salary
cat_eda_scripts(hr_data, "salary")

print('''
      Analyst Comments/Observations:
      Based on the metadata provided, salary is defined 
      as: "Relative level of salary (high)".
      ''')


# In[ ]:


#EDA for Number of Projects by Left
cat_by_cat_eda_scripts(hr_data, 'number_project', 'left')

print('''
      Analyst Comments/Observations:
      When grouping the number_projects by left, it appears that over 40% of the 
      people who left only completed 2 projects, while the rest of the people who left
      completed 4 or more projects. Compared to the people who did not leave
      most of the people completed 3 or 4 projects.
      ''')


# In[ ]:


#EDA for Sales by Left
cat_by_cat_eda_scripts(hr_data, 'sales', 'left')

print('''
      Analyst Comments/Observations:
      The relative distribution of the people who left compared to the 
      department they are in appears to be fairly evenly distributed 
      when comparing those who left with those who did not.
      ''')


# In[ ]:


#EDA for Salary by Left
cat_by_cat_eda_scripts(hr_data, 'salary', 'left')

print('''
      Analyst Comments/Observations:
      The employees who left had a higher percent of 'low salary' than those who 
      stayed and a lower percent of 'high salary' than those who stayed.
      ''')


# In[ ]:


#EDA for Promotion last 5 Years by Left
cat_by_cat_eda_scripts(hr_data, 'promotion_last_5years', 'left')

print('''
      Analyst Comments/Observations:
      Those who had a promotion in the last five year are slightly less likely to have left the company.
      ''')


# In[ ]:


#EDA for Work Accident by Left
cat_by_cat_eda_scripts(hr_data, 'Work_accident', 'left')

print('''
      Analyst Comments/Observations:
      Those who have not had a work accident are more likely to have left the company.
      ''')


# In[ ]:


#EDA for Satisfaction Level by Left
numeric_by_cat_eda(hr_data, 'satisfaction_level', 'left')

print('''
      Analyst Comments/Observations:
      The satisfation level of employees who stayed were typically between .55 and .85 (about) while those
      who left were below .70. Overal, those who left the company had a lower satisfaction level than those 
      who stayed.
      ''')


# In[ ]:


#EDA for Last Evaluation by Left
numeric_by_cat_eda(hr_data, 'last_evaluation', 'left')

print('''
      Analyst Comments/Observations:
      
      Those who left had, on average an evaluation further in the past than those who stayed though 
      the body of the distribution is larger also.
      ''')


# In[ ]:


#EDA for Average Monthly Hours by Left
numeric_by_cat_eda(hr_data, 'average_montly_hours', 'left')

print('''
      Analyst Comments/Observations:
      
      The employees wholeft had on average a higher number of hours per month than those who stayed.
      ''')


# In[ ]:


#Data Prep for Modeling

#Binarize the Class Variables
X = hr_data.drop('left',axis=1)
y = hr_data['left']
y = y.astype(int)
print(y.dtypes)

print(X.dtypes)
X = pd.get_dummies(X, prefix=['sales', 'salary'])
print()
print(X.head())

print()
print(X.dtypes)

#Create Train and Test Data Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print()
print("length of x = " + str(len(X)))
print("length of x train = " + str(len(X_train)))
print("length of x test = " + str(len(X_test)))

#Setup Kfold
kfold = KFold(n_splits=10, random_state=7)
scoring = 'accuracy'


# In[ ]:


#Logistic Regression
print("Logistic Regression")
print()

# Tunning the hyper paramters
tuned_parameters = [{'penalty': ['l1','l2'], 
                     'C': [1, 10, 100, 1000],
                     'class_weight': ['balanced',None]}]

scores = ['accuracy','precision', 'recall', 'f1']

grid_search(scores, LogisticRegression, tuned_parameters, "Logistic Regression Model")

print('''
    Analyst comments:
    The final hyper parameters where selected basaed on the recall parameters and are:
    {'class_weight': 'balanced', 'C': 1, 'penalty': 'l1'}
    
    The recall rate was 0.81  with these parameters.
''')


# In[ ]:


#Logistic Regression
print("Logistic Regression - Selected Hyper Parameters")
print()

#Fit the select hyper paramters
logreg = LogisticRegression(penalty='l1', C=1, class_weight='balanced')
logreg.fit(X_train, y_train)

train_accuracy = logreg.score(X_train, y_train)
test_accuracy = logreg.score(X_test, y_test)

#Calculate Out of Sample Predictions
y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)
 
#K Fold Validation 
results = cross_val_score(logreg, X, y, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))    
    
#Model Train    
classification_output_report(X_train, y_train, X_test, y_test, y_pred, y_pred_prob, train_accuracy, test_accuracy)

print('''
    Analyst Comments:
    The Kfold accuracy is .75 while the train and test accuracy's were both 0.76.
    The recall from the model is 0.81.
''')


# In[ ]:


#SVM
print("Support Vector Machine - Hyper Paramter Selection")
print()

# Tunning the hyper paramters
tuned_parameters = [{'dual':[False],
                     'penalty': ['l1','l2'], 
                     'C': [1, 10, 100, 1000],
                     'class_weight': ['balanced',None]}]

scores = ['accuracy','precision', 'recall', 'f1']

grid_search(scores, LinearSVC, tuned_parameters, "Linear SVM")

print('''
    Analyst comments:
    The final hyper parameters where selected basaed on the recall parameters and are:
    {'class_weight': 'balanced', 'C': 1, 'dual': False, 'penalty': 'l1'}
    
    The recall rate was 0.80  with these parameters.
''')


# In[ ]:


#SVM
print("Support Vector Machine - Selected Hyper Paramters")
print()

#Fit The Model
svm = LinearSVC(penalty='l1', dual=False, C=.5, class_weight='balanced')
svm.fit(X_train, y_train)

train_accuracy = svm.score(X_train, y_train)
test_accuracy = svm.score(X_test, y_test)

#Calculate Out of Sample Predictions
y_pred = svm.predict(X_test)
y_pred_prob = None #svm.predict_proba(X_test) does not exist

#K Fold Validation 
results = cross_val_score(svm, X, y, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
    
#Model Train    
classification_output_report(X_train, y_train, X_test, y_test, y_pred, y_pred_prob, train_accuracy, test_accuracy)

print('''
    Analyst Comments:
    The Kfold accuracy is .75 while the train and test accuracy's were 76 and 75 respectively.
    The recall from the model is 0.80.
''')


# In[ ]:


#Random Forest
print("Random Forest - Hyper Paramter Selection")
print()

# Tunning the hyper paramters
tuned_parameters = [{'n_estimators':[250,500],
                     'max_features': ['sqrt','log2'], 
                     'criterion': ['gini'],
                     'max_depth': [25,50,None],
                     'min_samples_split': [5,10],
                     'min_samples_leaf': [5,10],
                     'max_leaf_nodes': [None]}]

scores = ['accuracy','precision', 'recall', 'f1']

grid_search(scores, RandomForestClassifier, tuned_parameters, "Random Forest")

print('''
    Analyst comments:
    The final hyper parameters where selected basaed on the recall parameters and are:
    {'n_estimators': 250, 'max_depth': None, 'max_features': 'sqrt', 'criterion': 'gini', 
    'min_samples_split': 10, 'max_leaf_nodes': None, 'min_samples_leaf': 5}
    
    The recall rate was 0.91  with these parameters.
''')


# In[ ]:


#Random Forest
print("Random Forest - Selected Hyper Paramters")
print()

#Fit The Model
rf_clf = RandomForestClassifier(oob_score=True, n_jobs=2, n_estimators=250, 
                                        max_features='sqrt', criterion='gini', max_depth=None, 
                                        min_samples_split=10, min_samples_leaf=5, max_leaf_nodes=None)
rf_clf.fit(X_train, y_train)

train_accuracy = rf_clf.score(X_train, y_train)
test_accuracy = rf_clf.score(X_test, y_test)

#Calculate Out of Sample Predictions
y_pred = rf_clf.predict(X_test)
y_pred_prob = rf_clf.predict_proba(X_test)
    
#K Fold Validation 
results = cross_val_score(rf_clf, X, y, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))    
    
#Model Train    
classification_output_report(X_train, y_train, X_test, y_test, y_pred, y_pred_prob, train_accuracy, test_accuracy)

print('''
    Analyst Comments:
    The Kfold accuracy is .97 while the train and test accuracy's were 98 and 97 respectively.
    The recall from the model is 0.91.
''')


# In[ ]:


#Naive Bayes
#Fit The Model
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

print("Naive Bayes")
print()

train_accuracy = nb_clf.score(X_train, y_train)
test_accuracy = nb_clf.score(X_test, y_test)

#Calculate Out of Sample Predictions
y_pred = nb_clf.predict(X_test)
y_pred_prob = nb_clf.predict_proba(X_test)

#K Fold Validation 
results = cross_val_score(nb_clf, X, y, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))    
    
#Model Train    
classification_output_report(X_train, y_train, X_test, y_test, y_pred, y_pred_prob, train_accuracy, test_accuracy)

print('''
    Analyst Comments:
    The Kfold accuracy is .63 while the train and test accuracy's were 0.66 and 0.67 respectively.
    The recall from the model is 0.84.
''')


# In[ ]:


#Ensamble Model of Top Three
#Fit The Model
logreg = LogisticRegression(penalty='l1', C=1, class_weight='balanced')
gbt_clf = RandomForestClassifier(oob_score=True, n_jobs=2, n_estimators=250, 
                                        max_features='sqrt', criterion='gini', max_depth=None, 
                                        min_samples_split=10, min_samples_leaf=5, max_leaf_nodes=None)
nb_clf = GaussianNB()

ensamble = VotingClassifier(estimators=[('lr', logreg), ('gbt', gbt_clf), ("nb", nb_clf)], 
                         voting='hard')

ensamble.fit(X_train, y_train)
print("Ensamble Model")
print()

train_accuracy = ensamble.score(X_train, y_train)
test_accuracy = ensamble.score(X_test, y_test)

#Calculate Out of Sample Predictions
y_pred = ensamble.predict(X_test)
y_pred_prob = None #ensamble.predict_proba(X_test)

#K Fold Validation 
results = cross_val_score(ensamble, X, y, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))    
    
#Model Train     
classification_output_report(X_train, y_train, X_test, y_test, y_pred, y_pred_prob, train_accuracy, test_accuracy)

print('''
    Analyst Comments:
    The Kfold accuracy is .82 while the train and test accuracy's were 0.83 and 0.84 respectively.
    The recall from the model is 0.88.
''')


# In[ ]:


print('''
Summary and Model Recommendation:

    Based on the above analysis the data provided is judged to be acceptable and ready for modeling.
    
    Of the modeling algorithms tested (logistic regresion, SVM, random forest, naive bayes, ensamble of the above) the
    best model, based on the recall metric, is the random forest. The second best model is the ensamble model.
    
    Recall was selected as the primary metric of interest because false positive prediction (predicting someone who
    would leave would not) is a lower cost item than false negative (predicting someone as not leaving who does leave) 
    for the company. The cost to HR of doing an employee intervention is less than that of hiring and training a new employee.
''')

