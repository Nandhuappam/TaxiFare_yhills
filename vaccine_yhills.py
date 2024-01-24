#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.preprocessing import LabelEncoder


# In[4]:


from sklearn.metrics import accuracy_score, classification_report


# In[5]:


from sklearn.linear_model import LogisticRegression


# In[6]:


from sklearn.svm import SVC


# In[7]:


from sklearn.tree import DecisionTreeClassifier


# In[8]:


from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier


# In[9]:


data = pd.read_csv('h1n1_vaccine_prediction.csv')


# In[10]:


print(data.head())


# In[11]:


label_encoder = LabelEncoder()


# In[12]:


categorical_columns = ['race', 'sex', 'income_level', 'marital_status', 'housing_status', 'employment', 'census_msa']


# In[13]:


for column in categorical_columns:


# In[14]:


data[column] = label_encoder.fit_transform(data[column])


# In[15]:


X = data.drop('h1n1_vaccine', axis=1)


# In[16]:


y = data['h1n1_vaccine']


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Classification': SVC(),
    'Decision Tree Classification': DecisionTreeClassifier(),
    'Bagging Classification': BaggingClassifier(),
    'AdaBoost Classification': AdaBoostClassifier(),
    'Gradient Boost Classification': GradientBoostingClassifier(),
    'Random Forest Classification': RandomForestClassifier()
}


# In[19]:


for name, model in models.items():
    print(f"\n{name}:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


# In[20]:


accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)


# In[21]:


print(f"Accuracy: {accuracy:.4f}")


# In[22]:


print("Classification Report:\n", report)


# In[ ]:




