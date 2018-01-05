from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

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
    
    print('F1 Score:' + str(np.round(metrics.f1_score(	, 
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
						
						
def train_predict_evaluate_model_cfmatrix(classifier, 
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
	print()
	
	labels = list(set(test_labels))
	cm = metrics.confusion_matrix(test_labels, predictions, labels)
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
	
    return predictions
	
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