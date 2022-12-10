#!/usr/bin/env python
# coding: utf-8

# # Data Science
# #### By: Javier Orduz
# [license-badge]: https://img.shields.io/badge/License-CC-orange
# [license]: https://creativecommons.org/licenses/by-nc-sa/3.0/deed.en
# 
# [![CC License][license-badge]][license]  [![DS](https://img.shields.io/badge/downloads-DS-green)](https://github.com/Earlham-College/DS_Fall_2022)  [![Github](https://img.shields.io/badge/jaorduz-repos-blue)](https://github.com/jaorduz/)  ![Follow @jaorduc](https://img.shields.io/twitter/follow/jaorduc?label=follow&logo=twitter&logoColor=lkj&style=plastic)
# 

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# # Part I: Simpler Linear Regression. Knowing data

# Data source [0].

# In[2]:


df = pd.read_csv("titanic3.csv")


# In[3]:


df.head(5)


# In[6]:


df['sex'] = np.where(df['sex']=='female',1, 0)
print(df['sex'])


# In[92]:


df.head(5)


# In[93]:


df.describe()


# In[94]:


df.isnull().sum()


# In[95]:


print(sum(df['age'].isnull()))


# In[96]:


df['age'].mean()


# In[97]:


df['fare'].mean()


# In[98]:


df['age'].fillna(df['age'].mean(), inplace = True) 


# In[99]:


df['fare'].fillna(df['fare'].mean(), inplace = True) 


# In[100]:


df.isnull().sum()


# In[101]:


print(sum(df['age'].isnull()))


# In[102]:


df.groupby('sex').mean()


# In[103]:


df.groupby('sex')['age'].mean()


# In[104]:


for i in df.sex:
    if i == 'male':
        df.sex[i]==1
    else:
        df.sex[i]==2
print(df.sex)


# In[107]:


print(df.isna().sum().sum())


# In[106]:


x = (df.columns)

for i in ((x)):
    print(df[i].isnull())


# In[82]:


for i in ((x)):
    if i == "sex" or i == "pclassor" or i == "survivedor" or i == "sex" or i == "age" or i == "sibsp" or i == "parch" or i == "fare" or i == "body":
        mean_value=df[i].mean()
       
    print(mean_value)
# Replace NaNs in column S2 with the
# mean of values in the same column
    df[i].fillna(value=mean_value, inplace=True)
    print('Updated Dataframe:')
    print(df)
    

print(df.head(5))


# In[110]:


df.groupby('sex').mean()


# ##  Exercises

# 1. For Sex column, you must change to 1 if the passenger was female and 0 if they were male. 
# 1. There some missing data, find a function to find the missing information in each column.
# 1. If you found some missing data, then you have two options: 
#     - fill it in. Recommended
#     - Drop the rows with the missing values.
# 1. Print the null values for each columns.
# 1. Get the average value for those columns that have any missing values.
# 1. Remove the null values, and replace them with the mean value.
# 1. Print how many values are nulls.
# 1. Filter the database by each gender, and find the mean.
#     - You can filter using groupby method.
# 1. Submmit your report in Moodle. Template https://www.overleaf.com/read/xqcnnnrsspcp

# ## Versions

# In[20]:


from platform import python_version
print("python version: ", python_version())
get_ipython().system('pip3 freeze | grep qiskit')


# # References

# [0] data https://tinyurl.com/2m3vr2xp
# 
# [1] numpy https://numpy.org/
# 
# [2] scipy https://docs.scipy.org/
# 
# [3] matplotlib https://matplotlib.org/
# 
# [4] matplotlib.cm https://matplotlib.org/stable/api/cm_api.html
# 
# [5] matplotlib.pyplot https://matplotlib.org/stable/api/pyplot_summary.html
# 
# [6] pandas https://pandas.pydata.org/docs/
# 
# [7] seaborn https://seaborn.pydata.org/
# 

# In[ ]:




