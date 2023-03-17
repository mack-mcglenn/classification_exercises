#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
import numpy as np
import env
import pandas as pd
import matplotlib as plt
import seaborn as  sb
import scipy.stats as stats
from pydataset import data


# In[31]:


def connect(db):
    
    """This function will pull the information from my env file (username, password, host,
    database) to connect to Codeup's MySQL database"""
    
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'


# In[32]:



def get_titanic_data():
    
    """This function will confirm whether or not there is a local csv containing the titanic
    dataset. If no such csv exists locally, this function will pull the titanic data from
    Codeup's MySQL database and return as a dataframe based on the credentials provided in 
    the env.py file in use"""

    if os.path.isfile('titanic.csv'):
        df = pd.read_csv('titanic.csv', index_col=0)
    else:
        query = """select * from passengers"""
        connection = connect('titanic_db')
        df = pd.read_sql(query, connection)
        df.to_csv('titanic.csv')
    return df

    


# In[29]:


paint_me = get_titanic_data()


# In[30]:


os.path.exists('titanic.csv')


# In[33]:



def get_iris_data():
    
    """This function will check whether or not there is a csv for the iris data saved locally
    If no such csv exists locally, this function will pull the iris data from
    Codeup's MySQL database and return as a dataframe based on the credentials provided in 
    the env.py file in use"""
    
    if os.path.isfile('iris.csv'):
        df = pd.read_csv('iris.csv', index_col=0)
    else:
        query = """select * from measurements
        join species using (species_id)"""
        connection = connect('iris_db')
        df = pd.read_sql(query, connection)
        df.to_csv('iris.csv')
    return df


# In[34]:


sleepingsickness = get_iris_data()


# In[35]:


os.path.exists('iris.csv')


# In[67]:



def get_telco_data():
    """This function will check whether or not there is a csv for the telco data saved 
      locally. If no such csv exists locally, this function will pull the telco data from
    Codeup's MySQL database and return as a dataframe based on the credentials provided in 
    the env.py file in use"""

    if os.path.isfile('telco.csv'):
        df = pd.read_csv('telco.csv', index_col=0)
    else:
        query = """select * from customers
        join contract_types using (contract_type_id)
        join internet_service_types using (internet_service_type_id)
        join payment_types using (payment_type_id)"""
        connection = connect('telco_churn')
        df = pd.read_sql(query, connection)
        df.to_csv('telco.csv')
    return df


# In[70]:


telcodf = get_telco_data()


# In[72]:


os.path.exists('telco.csv')


# In[ ]:




