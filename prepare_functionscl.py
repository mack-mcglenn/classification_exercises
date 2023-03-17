#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import numpy as np
import env
import pandas as pd
import acquire_copy as acq
import matplotlib as plt
import seaborn as  sb
import scipy.stats as stats
from pydataset import data
from sklearn.model_selection import train_test_split


# In[33]:


iris = acq.get_iris_data()
titanic = acq.get_titanic_data()
telco = acq.get_telco_data()


# In[40]:


#Basic/General Split function

def tvt_split(df, target):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on 
    [target variable].
    return train, validate, test DataFrames.
    '''
    
    #Test set
    train_validate, test = train_test_split(df, test_size=.20, random_state=123, 
                                            stratify=df[target])
    #Final train/val set
    train, val = train_test_split(train_validate, test_size=.30, random_state=123,
                                 stratify=train_validate[target])
    
    return train,val,test


# In[43]:


# Iris DF Functions

def prep_iris(df):
    """This function preps data in the iris csv (via the get_iris_data() function
    in acquire_copy) for future use"""
    df = df.drop(columns=['species_id', 'measurement_id'])
    df = df.rename(columns={'species_name':'species'})
    df=  pd.concat([df, (pd.get_dummies(df['species']))], axis = 1)
    
    return df


def split_iris(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on species.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.species)
    return train, validate, test


# In[19]:


prep_iris(iris).head()


# In[45]:


# Titanic DF Functions

def prep_titanic(df):
    """This function preps data in the titanic csv (via the get_titanic_data() function
    in acquire_copy) for future use"""
    df=df.drop(columns=['embarked', 'class', 'age', 'deck'])
    dummy_df = pd.get_dummies(data=titanic[['sex','embark_town']], drop_first=True)
    df = pd.concat([titanic, dummy_df], axis = 1)
    
    return titanic


def split_titanic(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on titanic.survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.survived)
    return train, validate, test


# In[38]:


prep_titanic(titanic).head()


# In[47]:


# Prep Telco DF Functions

def prep_telco(telco):
    """This function presp data from the telco csv (acquired via the get_telco_data() function in 
    acquire_copy and preps it for future use."""
    
    # drop unnecessary/redundant columns
    telco=telco.drop(columns=['internet_service_type_id', 'payment_type_id', 'contract_type_id'])
    
    
    # convert total charges column from str to float
    telco['total_charges']= (telco['total_charges'] + '0').astype('float')
    
    # convert binary cat variables to numeric
    telco['churn_bin'] = telco['churn'].map({'Yes': 1, 'No': 0})
    telco['gender_bin'] = telco['gender'].map({'Female': 1, 'Male': 0})
    telco['partner_bin'] = telco['partner'].map({'Yes': 1, 'No': 0})
    telco['dependents_bin'] = telco['dependents'].map({'Yes': 1, 'No': 0})
    telco['paperless_billing_bin'] = telco['paperless_billing'].map({'Yes': 1, 'No': 0})
    telco['phone_service_bin'] = telco['phone_service'].map({'Yes': 1, 'No': 0})
#     telco['multiple_lines_bin'] = telco['multiple_lines'].str.replace('No phone service, '0').map({'Yes': 2, 'No': 1})
#     telco['tech_support_bin'] = telco['tech_support'].str.replace('No internet service, '0').map({'Yes': 2, 'No': 1})
#     telco['internet_service_type_bin'] = telco['internet_service_type'].str.replace('No internet service, '0').map({'Yes': 2, 'No': 1})
    
    # Dummy variables for enby cat variables
    radioshack= pd.get_dummies( telco[['multiple_lines',                                        'online_security',                                        'online_backup',                                        'device_protection',                                        'tech_suport',                                        'payment_type',                                        'streaming_tv',                                        'streaming_movies',                                       'internet_service_type',                                       'contract_type' 
                                      ]], drop_first= True)
    telco= pd.concat([telco, radioshack], axis=1)
    
    return telco

def split_telco(df):
    '''
    take in a DataFrame and splits on telco.churn.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.churn)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.churn)
    return train, validate, test

