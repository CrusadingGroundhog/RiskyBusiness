#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd downloads


# In[2]:


pwd


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


# ### Read the CSV into DataFrame

# In[5]:


# Load the data
df = pd.read_csv('lending_data.csv')
df.head()


# In[6]:


df=pd.get_dummies(df, columns=["homeowner"])


# ### Split the Data into Training and Testing

# In[7]:


# Create our features
x = df.drop(columns=["loan_status"])

# Create our target
y = pd.DataFrame(df["loan_status"])


# In[8]:


x.describe()


# In[9]:


# Check the balance of our target values
y['loan_status'].value_counts()


# In[14]:


# Create X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, stratify=y)
x_train.shape


# ### Data Pre-Processing
# Scale the training and testing data using the StandardScaler from sklearn. Remember that when scaling the data, you only scale the features data (X_train and X_testing).

# In[15]:


# Create the StandardScaler instance
from sklearn.preprocessing import StandardScaler
data_scaler = StandardScaler()


# In[16]:


# Fit the Standard Scaler with the training data
# When fitting scaling functions, only train on the training dataset
data_scaler.fit(x_train)


# In[17]:


# Scale the training and testing data
x_train = data_scaler.transform(x_train)
x_test = data_scaler.transform(x_test)


# ### Simple Logistic Regression

# In[18]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(x_train, y_train)


# In[19]:


# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(x_test)
balanced_accuracy_score(y_test, y_pred)


# In[20]:


# Display the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[21]:


# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))


# ### Oversampling
# 
# In this section, you will compare two oversampling algorithms to determine which algorithm results in the best performance. You will oversample the data using the naive random oversampling algorithm and the SMOTE algorithm. For each algorithm, be sure to complete the folliowing steps:
# 
#     View the count of the target classes using Counter from the collections library.
#     Use the resampled data to train a logistic regression model.
#     Calculate the balanced accuracy score from sklearn.metrics.
#     Print the confusion matrix from sklearn.metrics.
#     Generate a classication report using the imbalanced_classification_report from imbalanced-learn.
# 
# Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests
# 

# #### Naive Random Oversampling

# In[26]:


# Resample the training data with the RandomOversampler
from imblearn.over_sampling import RandomOverSampler

ROS = RandomOverSampler(random_state=1)
x_resampled, y_resampled = ROS.fit_resample(x_train, y_train)

# View the count of target classes with Counter
Counter(y_resampled)


# In[27]:


# Train the Logistic Regression model using the resampled data
ROS_model = LogisticRegression(solver='lbfgs', random_state=1)
ROS_model.fit(x_resampled, y_resampled)


# In[28]:


# Calculated the balanced accuracy score
ROS_y_pred = ROS_model.predict(x_test)
balanced_accuracy_score(y_test, ROS_y_pred)


# In[31]:


# Display the confusion matrix
confusion_matrix(y_test, ROS_y_pred)

## Briefly experimented for fun, there's a difference when you put y_test in front and when you put ROS_y_pred in front
### [622/3/111/18648] vs. [622/1111/3/18648]
#### Don't think there'd be a difference in outcome as long as I keep things the same throughout


# In[34]:


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, ROS_y_pred))


# ### SMOTE Oversampling

# In[35]:


# Resample the training data with SMOTE
from imblearn.over_sampling import SMOTE

x_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy=1.0).fit_resample(
    x_train, y_train
)
# View the count of target classes with Counter
Counter(y_resampled)


# In[36]:


# Train the Logistic Regression model using the resampled data
smote_model =  LogisticRegression(solver='lbfgs', random_state=1)
smote_model.fit(x_resampled, y_resampled)


# In[37]:


# Calculated the balanced accuracy score
smote_y_pred = smote_model.predict(x_test)
balanced_accuracy_score(y_test, smote_y_pred)


# In[38]:


# Display the confusion matrix
confusion_matrix(y_test, smote_y_pred)


# In[42]:


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, smote_y_pred))


# In[ ]:




