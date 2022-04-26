#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
df = pd.read_csv('CW2_Facebook_metrics _.csv')
df.head(10)


# In[37]:


df.isna().sum()


# In[38]:


df.isna().sum()/len(df)*100


# In[39]:


df.dropna(axis=1)


# In[40]:


df.dropna(subset=['Paid', 'Likes', 'Shares'],axis=0,inplace=True)


# In[41]:


df.isna().sum()/len(df)*100


# In[42]:


df.dropna(thresh=0.8*len(df),axis=1,inplace=True)


# In[43]:


#check dataset size
print(df.shape)


# In[44]:


#split data into inputs and targets
X = df.drop(columns = ['Category'])
y = df['Category']


# In[45]:


from sklearn.model_selection import train_test_split
#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, stratify=y)


# In[46]:


df.head()


# In[47]:



import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier() 
#create a dictionary of all values we want to test for n_neighbors
params_knn = {'n_neighbors': np.arange(1, 25)} 
#use gridsearch to test all values for n_neighbors
knn_gs = GridSearchCV(knn, params_knn, cv=5) 
#fit model to training data
knn_gs.fit(X_train, y_train)


# In[48]:


#save best model
knn_best = knn_gs.best_estimator_
#check best n_neigbors value
print(knn_gs.best_params_)


# In[49]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
#create a dictionary of all values we want to test for n_estimators
params_rf = {'n_estimators': [50, 100, 200]}
#use gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=5)#fit model to training data
rf_gs.fit(X_train, y_train)


# In[50]:


#save best model
rf_best = rf_gs.best_estimator_
#check best n_estimators value
print(rf_gs.best_params_)


# In[59]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
#fit the model to the training data
log_reg.fit(X_train, y_train)


# In[68]:


#test the three models with the test data and print their accuracyscores
print('knn: {}'.format(knn_best.score(X_test, y_test)))
print('rf: {}'.format(rf_best.score(X_test, y_test)))
print('log_reg: {}'.format(log_reg.score(X_test, y_test)))


# In[57]:


from sklearn.ensemble import VotingClassifier
#create a dictionary of our models
estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg)]
#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='hard')


# In[60]:


#fit model to training data
ensemble.fit(X_train, y_train)
#test our model on the test data
ensemble.score(X_test, y_test)


# In[71]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
sgd_model = SGDClassifier(random_state=42)
sgd_model.fit(X_train, y_train)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
y_pred_sgd = sgd_model.predict(X_test)
y_pred_log = log_model.predict(X_test)

from sklearn.metrics import accuracy_score
knn_score = accuracy_score(y_test, y_pred_knn)
sgd_score = accuracy_score(y_test, y_pred_sgd)
log_score = accuracy_score(y_test, y_pred_log)
print("Accuracy score (KNN): ", knn_score)
print("Accuracy score (SGD): ", sgd_score)
print("Accuracy score (Logistic): ", log_score)


# In[83]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
knn_pipe = Pipeline([('mms', MinMaxScaler()),('knn', KNeighborsClassifier())])
params = [{'knn__n_neighbors': [3, 5, 7, 9],'knn__weights': ['uniform', 'distance'],'knn__leaf_size': [15, 20]}]
gs_knn = GridSearchCV(knn_pipe, param_grid=params, scoring='accuracy', cv=5)
gs_knn.fit(X_train, y_train)
gs_knn.best_params_


# In[84]:


# find best model score
gs_knn.score(X_train, y_train)


# In[80]:


df.head()


# In[97]:


#Importing the required libraries
from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
 


# In[99]:


#Splitting the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
                                    df.iloc[:,:-1], df.iloc[:,-1], test_size=0.3, random_state=42)
 


# In[100]:


# Initialize and fit the Model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[101]:


#Make prediction on the test set
pred = model.predict(X_test)
 


# In[108]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[111]:


# Add noisy features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)


# In[ ]:




