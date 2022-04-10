#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install pandas')


# In[2]:


get_ipython().system('pip3 install graphviz')


# In[3]:


get_ipython().system('pip3 install scikit-learn')


# In[4]:


# Pakiet do plików xlsx
get_ipython().system('pip3 install openpyxl')


# In[6]:


import pandas as pd # load and manipulate data and for One-Hot Encoding
import numpy as np # calculate the mean and standard deviation
import xgboost as xgb # XGBoost stuff
from sklearn.model_selection import train_test_split # split  data into training and testing sets
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer # for scoring during cross validation
from sklearn.model_selection import GridSearchCV # cross validation
from sklearn.metrics import confusion_matrix # creates a confusion matrix
from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix


# In[7]:


# Wersje pakietów na przykładzie pandas
print(pd.show_versions())


# In[1]:


get_ipython().system('pip3 install matplotlib')


# In[2]:


get_ipython().system('pip3 install lightgbm')


# In[4]:


get_ipython().system('pip3 install seaborn')


# In[1]:


get_ipython().system('pip3 install sklearn')


# In[2]:


get_ipython().system('pip3 install boruta')

