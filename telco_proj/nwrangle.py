



import os
import env
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as  sb
import scipy.stats as stats
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#Acquire Functions
def connect(db):
    
    """This function will pull the information from my env file (username, password, host,
    database) to connect to Codeup's MySQL database"""
    
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_telco_data():
    """This function will check whether or not there is a csv for the telco data saved 
      locally. If no such csv exists locally, this function will pull the telco data from
    Codeup's MySQL database and return as a dataframe based on the credentials provided in 
    the env.py file in use"""

    if os.path.isfile('telco.csv'):
        telco = pd.read_csv('telco.csv', index_col=0)
    else:
        query = """select * from customers
        join contract_types using (contract_type_id)
        join internet_service_types using (internet_service_type_id)
        join payment_types using (payment_type_id)"""
        connection = connect('telco_churn')
        telco = pd.read_sql(query, connection)
        telco.to_csv('telco.csv')
    return telco


# Data Dictionary Functions
def sort_cols(telco):
    """ Allows me to sort columns alhpabetically by type(obj, float, int) for my data dictionary"""

    grouped_cols = {'float_cols': [], 'int_cols': [], 'obj_cols': []}
    for col in telco.columns:
        if telco[col].dtype == 'float':
            grouped_cols['float_cols'].append(col)
        elif telco[col].dtype == 'int':
            grouped_cols['int_cols'].append(col)
        else:
            grouped_cols['obj_cols'].append(col)
            
    return telco.columns

sorted_cols = {}
grouped_cols = {'float_cols': [], 'int_cols': [], 'obj_cols': []}
for col_type, cols in grouped_cols.items():
    sorted_cols[col_type] = sorted(cols)

 
data_dict1 = pd.read_csv('Telcotargetdict.csv', index_col=0)
data_dict2 = pd.read_csv(' Telcodict.csv', index_col=0)




#Prepare Functions
def prep_df(df):
    """This (revised) function preps data from the df csv (acquired via the get_df_data() function in 
    acquire_copy and preps it for future use."""
    
    # drop unnecessary/redundant columns
    df.drop(columns=['internet_service_type_id', 'payment_type_id', 'contract_type_id'])
    
    # convert binary cat variables to numeric
    df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
    df['gender_bin'] = df['gender'].map({'Female': 1, 'Male': 0})
    df['partner_bin'] = df['partner'].map({'Yes': 1, 'No': 0})
    df['dependents_bin'] = df['dependents'].map({'Yes': 1, 'No': 0})
    df['paperless_billing_bin'] = df['paperless_billing'].map({'Yes': 1, 'No': 0})
    df['phone_service_bin'] = df['phone_service'].map({'Yes': 1, 'No': 0})
    
    # Dummy variables for enby cat variables
    radioshack= pd.get_dummies( df[[   'tech_support', \
                                       'multiple_lines', \
                                       'tech_support', \
                                       'payment_type', \
                                       'internet_service_type',\
                                       'contract_type',\
                                      ]], drop_first= False)
    df= pd.concat([df, radioshack], axis=1)
    
    return df 

                                       
   #Train Test Split 
def tvt_split(df, target):
     # 80% train_validate, 20% test
# then of the 80% train_validate: 30% validate, 70% train.
# Final split will be 60/20/20 
  
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, val = train_test_split(train, test_size=.30, random_state=123, stratify=train[target])

    return train,val,test

# MODELING

def model_setup(train,val,test):
    """ This function prepares data for modeling by splitting the variables I want to use for modeling into  X_train, X_val, X_test and y_train, y_val, y_test"""
    
# Disregard
# My month-to-month contract variable wasn't specifically identified during the concatenation in the
 # # prep phase. I'll identify that now and assign month-to-month contracts as integers
 #    train['contract_type_m2m'] = train.contract_type == 'Month-to-month'
 #    train['contract_type_m2m']= train['contract_type_m2m'].astype(int)
    
 # Identify the categories kept for modeling
    keep_cats = ['churn', 
            'tenure',
            'dependents_bin',
            'partner_bin',
            'internet_service_type_Fiber optic',
            'monthly_charges',
            'multiple_lines_Yes',
            'tech_support_Yes',
            'contract_type_Month-to-month']
 
 # Reassign train/val/test variables
    train=train[keep_cats]
    val=val[keep_cats]
    test=test[keep_cats]

 # Split model variables, x= predict, y= target
    X_train= train.drop(columns='churn').reset_index(drop= True)
    X_val=val.drop(columns='churn').reset_index(drop= True)
    X_test=test.drop(columns='churn').reset_index(drop= True)
    y_train= train.churn
    y_val= val.churn
    y_test= test.churn
    
    return X_train, y_train, X_val, y_val, X_test, y_test


#____Decision Trees____


def df_trval_model(X_train, y_train, X_val, y_val):
    dt_val = DecisionTreeClassifier(max_depth=5)

    # fitting the KNN classifier with training data
    dt_val.fit(X_train, y_train)

    # predicting churn outcome for test data
    y_pred = dt_val.predict(X_val)

    # model score/Val accuracy
    model_score = dt_val.score(X_val, y_val)

    # confusion matrix
    confusion_mat = confusion_matrix(y_val, y_pred)

    # getting the classification report
    classification_rep = classification_report(y_val, y_pred)
    
    # plot the thing
    plot_tree(dt_val, feature_names= X_train.columns, class_names=['0','1'])

    print(f'Accuracy of Decision Tree on train data: {dt_val.score(X_train, y_train)}')
    print(f'Model score/ Accuracy of Decision Tree on val data: {model_score}')
    print("\nClassification Report:\n", classification_rep)
    print("\nConfusion Matrix:\n", confusion_mat)
  
    
def df_trtest_model(X_train, y_train, X_test, y_test):
    dt_test = DecisionTreeClassifier(max_depth=5)

    # fitting the KNN classifier with training data
    dt_test.fit(X_train, y_train)

    # predicting churn outcome for test data
    y_pred = dt_test.predict(X_test)

    # model score/Val accuracy
    model_score = dt_test.score(X_test, y_test)

    # confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    # getting the classification report
    classification_rep = classification_report(y_test, y_pred)
    
    # plot the thing

    print(export_text(dt_test, feature_names=X_train.columns.tolist()))
    print(f'Accuracy of DecisionTree on train data: {dt_test.score(X_train, y_train)}')
    print(f'Model score/ Accuracy of DecisionTree on val data: {model_score}')
    print("\nClassification Report:\n", classification_rep)
    print("\nConfusion Matrix:\n", confusion_mat)
    

def dt_change_depth(X_train, y_train):
    """ This function takes in a target (y_train) and train dataset sans target (X_train)
    and returns a dataframe that runs a Decision Tree model with multiple depths"""
    # christmas={}
    for i in range(1,9):
     #make the thing  
        tree = DecisionTreeClassifier(max_depth= i, random_state= 123)
     # fit the thing
        tree.fit(X_train, y_train)
     # predict the thing
        y_pred= tree.predict(X_train)
     # confusion matrix
        mr_smith= confusion_matrix(y_train, y_pred)
     # Assigning variables for confusion matrix
        TN, FP, FN, TP = mr_smith[1,1], mr_smith[0,1], mr_smith[1,0], mr_smith[0,0]

     # Assigning variables for evaluation

        All = TN + FP+ FN+ TP

        accuracy= (TP + TN) / All
        True_pos_rate = recall= TP/ (TP + FN)
        precision = TP/ (TP + FP)
        support_pos = TP + FN
        support_neg = FP +TN
        classrep_dts= pd.DataFrame(classification_report(y_train, y_pred, output_dict=True))
        print(f'Decision Tree depth: {i}')
        print(classrep_dts)
        
def dt__comp_train_test(X_train, y_train): 
    christmas=[]
    for i in range(1,9):
        # make the thing
        tree=DecisionTreeClassifier(max_depth= i, random_state= 123)
        # fit the thing 
        tree.fit(X_train, y_train)
        # use the thing to evaluate model performance
        out_of_sample= tree.score(X_test, y_test)
        in_sample=tree.score(X_train, y_train)
        difference= in_sample - out_of_sample
        #labeling columns for table
        heads= {'max_depth': {i}, 
                'train_accuracy' : in_sample,
                'test_accuracy' : out_of_sample,
                'train_test_difference' : difference}
        christmas.append(heads)
    willow = pd.DataFrame(christmas)
    return willow

def dt_comp_train_val(X_train, y_train, X_val, y_val):
    # Make some slight variations to the function I used earlier
    christmas=[]
    for i in range(1,9):
        # make the thing
        tree=DecisionTreeClassifier(max_depth= i, random_state= 123)
        # fit the thing 
        tree.fit(X_train, y_train)
        # use the thing to evaluate model performance
        out_of_sample= tree.score(X_val, y_val)
        in_sample=tree.score(X_train, y_train)
        difference= round((in_sample - out_of_sample) * 100, 2)
        #labeling columns for table
        heads= {'max_depth': {i}, 
                'train_accuracy' : in_sample,
                'val_accuracy' : out_of_sample,
                'train_val_difference' : difference}
        christmas.append(heads)
    willow = pd.DataFrame(christmas)
    return willow
# ___________Random Forest__________
def rf_multi_test(X_train, y_train, X_test, y_test):
    """This function takes in the train and test datasets and computes their respective
    accuracy scores and the difference between those scores when the max_depth and min_samples
    are changed"""
    little_john=[]
    # set range for max_Depth starting at 1, up to 15, counting by 2
    for i in range(1,15,2):
    # set range forin_samples starting at 3, up to 20, counting by 3
        for x in range(3,20,3):
    # fit a Random Forest classifier
            sherwood = RandomForestClassifier(max_depth= i, min_samples_leaf= x, random_state=123)

            rftestfit = sherwood.fit(X_train, y_train)

    # make predictions on the test set
            rftest_pred = sherwood.predict(X_train)

    # calculate model scores
            test_score = sherwood.score(X_test, y_test)
            train_score= sherwood.score(X_train, y_train)
            difference = round((train_score - test_score) * 100, 2)

            labels = {'max_depth': i,
                           'min_samples_leaf': x,
                           'Train Accuracy': train_score,
                           'Test Accuracy': test_score,
                           'Percentage Difference': difference
                           }
    # create df that measures train score, test score, and the difference between them
            little_john.append(labels)
    return pd.DataFrame(little_john)

def rf_trtest_model(X_train, y_train, X_test, y_test):
    """Takes my best performing RF model based on max depth and leaves, and return confusion matrix
        and classification report"""
    rf_test = RandomForestClassifier(max_depth=5, min_samples_leaf= 3, random_state=123)

    # fitting the RF classifier with training data
    rf_test.fit(X_train, y_train)

    # predicting churn outcome for test data
    y_pred = rf_test.predict(X_test)

    # model score/Val accuracy
    model_score = rf_test.score(X_test, y_test)

    # confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    # getting the classification report
    classification_rep = classification_report(y_test, y_pred)

    print(f'Accuracy of Random Forest on train data: {rf_test.score(X_train, y_train)}')
    print(f'Model score/ Accuracy of Random Forest on test data: {model_score}')
    print("\nClassification Report:\n", classification_rep)
    print("\nConfusion Matrix:\n", confusion_mat)
def rf_trval_model(X_train, y_train, X_val, y_val):
    """Takes my best performing RF model based on max depth and leaves, and return confusion matrix
        and classification report"""
    rf_val = RandomForestClassifier(max_depth=5, min_samples_leaf= 6, random_state=123)

    # fitting the RF classifier with training data
    rf_val.fit(X_train, y_train)

    # predicting churn outcome for test data
    y_pred = rf_val.predict(X_val)

    # model score/Val accuracy
    model_score = rf_val.score(X_val, y_val)

    # confusion matrix
    confusion_mat = confusion_matrix(y_val, y_pred)

    # getting the classification report
    classification_rep = classification_report(y_val, y_pred)

    print(f'Accuracy of Random Forest on train data: {rf_val.score(X_train, y_train)}')
    print(f'Model score/ Accuracy of Random Forest on validate data: {model_score}')
    print("\nClassification Report:\n", classification_rep)
    print("\nConfusion Matrix:\n", confusion_mat)
    
    
def rt_multi_val(X_train, y_train, X_val, y_val):
    """This function takes in the train and test datasets and computes their respective
    accuracy scores and the difference between those scores when the max_depth and min_samples
    are changed"""
    little_john=[]
    # set range for max_Depth starting at 1, up to 15, counting by 2
    for i in range(1,15,2):
    # set range forin_samples starting at 3, up to 20, counting by 3
        for x in range(3,20,3):
    # fit a Random Forest classifier
            sherwood = RandomForestClassifier(max_depth= i, min_samples_leaf= x, random_state=123)

            rftestfit = sherwood.fit(X_train, y_train)

    # make predictions on the test set
            rftest_pred = sherwood.predict(X_train)

    # calculate model scores
            val_score = sherwood.score(X_val, y_val)
            train_score= sherwood.score(X_train, y_train)
            difference = round((train_score - val_score) * 100, 2)

            labels = {'max_depth': i,
                           'min_samples_leaf': x,
                           'Train Accuracy': train_score,
                           'Validate Accuracy': val_score,
                           'Percentage Difference': difference
                           }
    # create df that measures train score, test score, and the difference between them
            little_john.append(labels)
    return pd.DataFrame(little_john)


def rf_trainval(X_train, y_train, X_val, y_val):
    
    # fit a Random Forest classifier
    sherwood = RandomForestClassifier(max_depth= 5, min_sample_leaf= 6, random_state=123)
    sherwood.fit(X_train, y_train)
    
    # make predictions on the test set
    y_pred = sherwood.predict(X_train)
    
    # calculate confusion matrix and classification report for non-churn customers
    confusion = confusion_matrix(non_churn_y_val, non_churn_y_pred)
    report = classification_report(non_churn_y_val, non_churn_y_pred)
    
    # calculate model score
    vscore = sherwood.score(X_val, y_val)
    trscore =sherwood.score(X_train, y_train)
    
    # return the results
    print(f'Accuracy of Random Forest on train data: {trscore}')
    print(f'Accuracy of Random Forest on val data: {vscore}')
    print(f'Confusion matrix: {confusion}')
    print(f'Classification Report: {report}')
    
# ______________KNN_________________________    
def knn_train(X_train, y_train):
    """This function uses KNN classifier and returns a classification report for in-sample data 
    with a range of neighbors"""
    # empty list for storing results
    results = []

    # loop through neighbor values and fit models
    for k in range(5, 18):
        # make the thing
        knn = KNeighborsClassifier(n_neighbors=k)
        # fit the thing
        knn.fit(X_train, y_train)
        # predict the thing
        y_pred = knn.predict(X_train)
        train_result = classification_report(y_train, y_pred, output_dict=True)
        results.append(train_result)

    # classification reports
    for i, r in enumerate(results):
        print(f"Model with {i+5} neighbors:")
        print(classification_report(y_train, y_pred))
        print("---------------")

        
def knn_multi_test(X_train, y_train, X_test, y_test):
    """This function takes in the train and test datasets and computes their respective
    accuracy scores and the difference between those scores when the number of neighbors is
    changed"""
    wont_you_be=[]
    # set range for n_neighbors starting at 5, up to 20
    for i in range(5,20):
    # fit a Random Forest classifier
            nextdoor = KNeighborsClassifier(n_neighbors= i)

            knnfit = nextdoor.fit(X_train, y_train)

    # make predictions on the test set
            y_pred = nextdoor.predict(X_train)

    # calculate model scores
            test_score = nextdoor.score(X_test, y_test)
            train_score= nextdoor.score(X_train, y_train)
            difference = round((train_score - test_score) * 100, 2)

            labels = {'n_neighbors': i,
                           'Weight': x,
                           'Train Accuracy': train_score,
                           'Test Accuracy': test_score,
                           'Percentage Difference': difference
                           }
    # create df that measures train score, test score, and the difference between them
            wont_you_be.append(labels)
    return pd.DataFrame(wont_you_be)

def mr_rogerstest(X_train, y_train, X_test, y_test):
    """using KNearest Neighbor to return accuracy of my target prediction on train and test sets
    """
# creating KNN classifier with number of neighbors=5
    knn_classifier = KNeighborsClassifier(n_neighbors = 15)

# fitting the KNN classifier with training data
    knn_classifier.fit(X_train, y_train)

# predicting churn outcome for test data
    y_pred = knn_classifier.predict(X_test)

# getting the model score
    model_score = knn_classifier.score(X_test, y_test)

# getting the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

# getting the classification report
    classification_rep = classification_report(y_test, y_pred)
    
    print(f'Accuracy of KNN on train data: {knn_classifier.score(X_train, y_train)}')
    print(f'Model score/ Accuracy of KNN on test data: {model_score}')
    # print("Model Score: ", model_score)
    print("\nClassification Report:\n", classification_rep)
    print("\nConfusion Matrix:\n", confusion_mat)

    
def knn_multi_val(X_train, y_train, X_val, y_val):
    """This function takes in the train and test datasets and computes their respective
    accuracy scores and the difference between those scores when the number of neighbors is
    changed"""
    wont_you_be=[]
    # set range for n_neighbors starting at 5, up to 20
    for i in range(5,20):
    # fit a Random Forest classifier
            nextdoor = KNeighborsClassifier(n_neighbors= i)

            knnfit = nextdoor.fit(X_train, y_train)

    # make predictions on the test set
            y_pred = nextdoor.predict(X_train)

    # calculate model scores
            val_score = nextdoor.score(X_val, y_val)
            train_score= nextdoor.score(X_train, y_train)
            difference = round((train_score - val_score) * 100, 2)

            labels = {'n_neighbors': i,
                           'Weight': x,
                           'Train Accuracy': train_score,
                           'Validate Accuracy': val_score,
                           'Percentage Difference': difference
                           }
    # create df that measures train score, test score, and the difference between them
            wont_you_be.append(labels)
    return pd.DataFrame(wont_you_be)

def mr_rogersval(X_train, y_train, X_val, y_val):
# creating KNN classifier with number of neighbors=5
    knn_classifier = KNeighborsClassifier(n_neighbors = 24)

# fitting the KNN classifier with training data
    knn_classifier.fit(X_train, y_train)

# predicting churn outcome for test data
    y_pred = knn_classifier.predict(X_val)

# model score/Val accuracy
    model_score = knn_classifier.score(X_val, y_val)

# confusion matrix
    confusion_mat = confusion_matrix(y_val, y_pred)

# getting the classification report
    classification_rep = classification_report(y_val, y_pred)
    
    print(f'Accuracy of KNN on train data: {knn_classifier.score(X_train, y_train)}')
    print(f'Model score/ Accuracy of KNN on val data: {model_score}')
    # print("Model Score: ", model_score)
    print("\nClassification Report:\n", classification_rep)
    print("\nConfusion Matrix:\n", confusion_mat)

    
# # def dt_change_depth(X_train, y_train):
# #     """ This function takes in a target (y_train) and train dataset sans target (X_train)
# #     and returns a dataframe that runs a Decision Tree model with multiple depths"""
#     # christmas={}
# for i in range(1,9):
#     for x in range(1,9)
# # fit a Random Forest classifier
# sherwood = RandomForestClassifier(max_depth= i, min_samples_leaf= x, random_seed=123)

# m2fit = sherwood.fit(X_train, y_train)
    
# # make predictions on the test set
# m2y_pred = sherwood.predict(X_test)

# # calculate model score
# score = sherwood.score(X_test, y_test)

#  report=

#     All = TN + FP+ FN+ TP

#     accuracy= (TP + TN) / All
#     True_pos_rate = recall= TP/ (TP + FN)
#     precision = TP/ (TP + FP)
#     support_pos = TP + FN
#     support_neg = FP +TN
#     classrep_dts= pd.DataFrame(classification_report(y_train, y_pred, output_dict=True))
#     print(f'Decision Tree depth: {i}')
#     print(classrep_dts)