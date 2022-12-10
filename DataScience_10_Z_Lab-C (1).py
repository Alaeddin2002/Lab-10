#!/usr/bin/env python
# coding: utf-8

# # Data Science
# #### By: Javier Orduz
# [license-badge]: https://img.shields.io/badge/License-CC-orange
# [license]: https://creativecommons.org/licenses/by-nc-sa/3.0/deed.en
# 
# [![CC License][license-badge]][license]  [![DS](https://img.shields.io/badge/downloads-DS-green)](https://github.com/Earlham-College/DS_Fall_2022)  [![Github](https://img.shields.io/badge/jaorduz-repos-blue)](https://github.com/jaorduz/)  ![Follow @jaorduc](https://img.shields.io/twitter/follow/jaorduc?label=follow&logo=twitter&logoColor=lkj&style=plastic)
# 

# In[10]:


import matplotlib.pyplot as plt
import math
import pandas as pd
import pylab as pl
import numpy as np
import io
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df = pd.read_csv("yelp.csv",encoding = 'latin1')


# In[12]:


t = df['text']
print(t[0].split())


# In[13]:


df.dtypes
cdf = df[['stars','cool','useful','funny']]


# In[14]:


msk = np.random.rand(len(df)) < 0.8
set_1 = cdf[msk]
set_2 = cdf[~msk]


# In[15]:


from sklearn import linear_model
regr = linear_model.LinearRegression()


# In[16]:


cdf.groupby("stars").mean()


# In[17]:


from sklearn.metrics import mean_squared_error
training_data = cdf.sample(frac=0.5, random_state=25)
testing_data = cdf.drop(training_data.index)
MSE = mean_squared_error(training_data, testing_data)
 
RMSE = math.sqrt(MSE)
print(RMSE)


# In[18]:


df.dtypes
cdf = df[['useful','funny']]
training_data = cdf.sample(frac=0.5, random_state=25)
testing_data = cdf.drop(training_data.index)
MSE = mean_squared_error(training_data, testing_data)
 
RMSE = math.sqrt(MSE)
print(RMSE)


# # Exercises

# 1. 1. Read `yelp.csv` into a DataFrame.
# 1. Evaluate the model by splitting it into training and testing sets and computing the RMSE. Does the RMSE make intuitive sense to you?
# 6. Try removing some of the features and see if the Root-Mean-Squared Error (RMSE) improves.
# 7. **Bonus:** Investigate of some new features you could create from the existing data that might be predictive of the response. (This is called "feature engineering".) Figure out how to create those features in Pandas, add them to your model, and see if the RMSE improves.
# 1. Submmit your report in Moodle. Template https://www.overleaf.com/read/xqcnnnrsspcp
# 
# <!---
# 8. **Bonus:** Compare your best RMSE on testing set with the RMSE for the "null model", which is the model that ignores all features and simply predicts the mean rating in the training set for all observations in the testing set.
# 9. **Bonus:** Instead of treating this as a regression problem, treat it as a classification problem and see what testing accuracy you can achieve with KNN.
# 10. **Bonus:** Figure out how to use linear regression for classification, and compare its classification accuracy to KNN.
# 1. Submmit your report in Moodle. Template https://www.overleaf.com/read/xqcnnnrsspcp
# --->

# ## Versions

# In[ ]:


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




