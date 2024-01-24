#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


from sklearn.svm import SVR


# In[21]:


from sklearn.tree import DecisionTreeRegressor


# In[22]:


from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor


# In[23]:


from sklearn.metrics import mean_squared_error


# In[24]:


from math import sqrt


# In[25]:


file_path = "taxifare_dataset.csv"


# In[26]:


df = pd.read_csv(file_path)


# In[27]:


df.head()


# In[28]:


X = df.drop(['amount'], axis=1)


# In[29]:


y = df['amount']


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):


# In[32]:


model.fit(X_train, y_train)


# In[33]:


y_pred = model.predict(X_test)


# In[34]:


rmse = sqrt(mean_squared_error(y_test, y_pred))


# In[35]:


print(f"{model_name} RMSE: {rmse}")


# In[36]:


linear_reg_model = LinearRegression()


# In[37]:


evaluate_model(linear_reg_model, X_train, y_train, X_test, y_test, "Linear Regression")


# In[38]:


svr_model = SVR()


# In[39]:


evaluate_model(svr_model, X_train, y_train, X_test, y_test, "Support Vector Regression")


# In[40]:


dt_model = DecisionTreeRegressor()


# In[41]:


evaluate_model(dt_model, X_train, y_train, X_test, y_test, "Decision Tree Regression")


# In[42]:


bagging_model = BaggingRegressor()


# In[43]:


evaluate_model(bagging_model, X_train, y_train, X_test, y_test, "Bagging Regression")


# In[44]:


adaboost_model = AdaBoostRegressor()


# In[45]:


evaluate_model(adaboost_model, X_train, y_train, X_test, y_test, "AdaBoost Regression")


# In[46]:


gradboost_model = GradientBoostingRegressor()


# In[47]:


evaluate_model(gradboost_model, X_train, y_train, X_test, y_test, "Gradient Boosting Regression")


# In[48]:


rf_model = RandomForestRegressor()


# In[49]:


evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest Regression")


# In[50]:


def haversine_distance(lat1, lon1, lat2, lon2):


# In[51]:


dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c


# In[52]:


return distance


# In[ ]:




