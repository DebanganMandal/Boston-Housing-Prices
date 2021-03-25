#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import missingno
import seaborn as sns
import sklearn.metrics as sm


# In[2]:


dataset=pd.read_csv("HousingData.csv")


# In[3]:


dataset.head(15)


# In[4]:


dataset.describe()


# In[5]:


dataset.info()


# In[6]:


missingno.matrix(dataset, figsize=(30,10))


# In[7]:


dataset.hist(bins=50, figsize=(20,15))
plt.show()


# In[8]:


corr_matrix = dataset.corr()
corr_matrix["MEDV"].sort_values(ascending=False)


# In[9]:


plt.figure(figsize=(20,15))
sns.heatmap(data=dataset.corr().abs(), annot=True)


# In[10]:


fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs=axs.flatten()
for k,v in dataset.items():
    sns.boxplot(y=k, data=dataset, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[98]:


dataset


# In[11]:


X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values
print(X)
print(y)


# In[13]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])
print(X)


# In[14]:


dataset.MEDV.isnull().sum() #So y has no null value


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[16]:


from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
X_train=mm.fit_transform(X_train)
X_test=mm.transform(X_test)
print(X_train)
print(X_test)


# In[76]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Successfull regression")
print("R2 Score = ", sm.r2_score(y_test, y_pred))
acc_lr = sm.r2_score(y_test, y_pred)


# In[77]:


from sklearn.linear_model import Ridge
regressor = Ridge(alpha = 1.0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Regression done")
print("R2 Score = ", sm.r2_score(y_test, y_pred))
acc_rr = sm.r2_score(y_test, y_pred)


# In[78]:


from sklearn.linear_model import Lasso
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X_train, y_train)
y_pred = lassoreg.predict(X_test)
print("Regression done")
print("R2 Score = ",r2_score(y_test, y_pred))
acc_lar = sm.r2_score(y_test, y_pred)


# In[79]:


from sklearn.svm import SVR
regressor_s = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Regression done")
print("R2 Score = ", sm.r2_score(y_test, y_pred))
acc_svr = sm.r2_score(y_test, y_pred)


# In[80]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Regression done")
print("R2-Score = ",r2_score(y_test, y_pred))
acc_dtr = sm.r2_score(y_test, y_pred)


# In[81]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Regression Done")
print("R2 Score = ",r2_score(y_test, y_pred))
acc_rfr = sm.r2_score(y_test, y_pred)


# In[82]:


models = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
              'Support Vector Regression', 'Decision Tree Regression', 
              'Random Forest Regression'],
    'Score': [
        acc_lr, 
        acc_rr,  
        acc_lar, 
        acc_svr, 
        acc_dtr, 
        acc_rfr,
    ]})
print("---Reuglar Accuracy Scores---")
models.sort_values(by='Score', ascending=False)


# In[ ]:




