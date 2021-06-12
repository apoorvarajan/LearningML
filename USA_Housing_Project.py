#!/usr/bin/env python
# coding: utf-8

# In[2]:


#First learning Project
#USA Housing 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('USA_Housing.csv')
df.head()


# In[6]:


df.corr().Price.sort_values()


# In[11]:


X=df[['Avg. Area Income','Avg. Area House Age','Area Population','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms']]
y=df.Price
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X_train,y_train)
pred1 = model1.predict(X_test)
plt.scatter(pred1,y_test)


# In[12]:


from sklearn import metrics
print("mean absolute error = ", metrics.mean_absolute_error(y_test,pred1))
print("mean squared error = ", metrics.mean_squared_error(y_test,pred1))
print("root mean squared error = ", np.sqrt(metrics.mean_squared_error(y_test,pred1)))


# In[ ]:




