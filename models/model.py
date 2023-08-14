#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pickle
import gzip
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


np.random.seed()


# In[11]:


# Exploratory data analysis


# In[12]:


data = pd.read_csv('./cardio_data.csv', sep=';', index_col='id')


# In[13]:


data.head()


# In[14]:


data.describe().transpose()


# In[15]:


data.isnull().sum()


# In[16]:


data['age'] = data['age'] // 365


# In[22]:


# Train test split


# In[23]:


X = data.drop(['cardio', 'age_group'], axis=1).values
y = data['cardio'].values


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[25]:


# Creating the model


# In[26]:


scaler = StandardScaler()


# In[27]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[28]:


model = RandomForestClassifier(n_estimators=300)


# In[29]:


model.fit(X_train, y_train)


# In[30]:


# Evaluating the model performance


# In[31]:


predictions = model.predict(X_test)


# In[32]:


accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.4f}')
confusion_matrix = confusion_matrix(y_test, predictions)
print('Confusion Matrix: ')
print(confusion_matrix, end='\n')
classification_report = classification_report(y_test, predictions)
print('Classification Report: ')
print(classification_report, end='\n')


# In[39]:


with gzip.open('model.pkl.gz', 'wb') as file:
    pickle.dump(model, file)


# In[ ]:




