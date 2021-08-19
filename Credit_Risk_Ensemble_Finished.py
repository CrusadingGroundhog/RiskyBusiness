#!/usr/bin/env python
# coding: utf-8

# ### Ensemble Learning

# In[1]:


cd downloads


# In[2]:


pwd downloads


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


# In[5]:


from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced


# ## Read CSV and Perform Basic Data Cleaning

# In[6]:


columns = [
    "loan_amnt", "int_rate", "installment", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "loan_status",
    "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
    "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag"
]

target = ["loan_status"]


# In[7]:


# Load the data

df = pd.read_csv('LoanStats_2019Q1.csv')

df.head()


# In[8]:


# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')

# Drop the null rows
df = df.dropna()


# ### Split the Data into Training and Testing

# In[9]:


# Create our features
X = df.drop(columns="loan_status")
X = pd.get_dummies(X, columns=['home_ownership', 'verification_status', 'issue_d', 'pymnt_plan', 'initial_list_status', 'next_pymnt_d', 'application_type', 'hardship_flag', 'debt_settlement_flag'])
# Create our target
y = df['loan_status'].to_frame()


# In[10]:


X.describe(include='all')


# In[11]:


# Check the balance of our target values
y['loan_status'].value_counts()


# In[12]:


# Create X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

X_train.shape


# ### Data-Pre Processing

# In[13]:


# Create the StandardScaler instance
from sklearn.preprocessing import StandardScaler
data_scaler = StandardScaler()


# In[14]:


# Fit the Standard Scaler with the training data
# When fitting scaling functions, only train on the training dataset
data_scaler.fit(X_train)


# In[15]:


# Scale the training and testing data
X_train = data_scaler.transform(X_train)
X_test = data_scaler.transform(X_test)


#  ### Ensemble Learners
# In this section, you will compare two ensemble algorithms to determine which algorithm results in the best performance. You will train a Balanced Random Forest Classifier and an Easy Ensemble classifier . For each algorithm, be sure to complete the folliowing steps:
# 
# 1.Train the model using the training data. 
# 
# 
# 2.Calculate the balanced accuracy score from sklearn.metrics.
# 
# 
# 3.Display the confusion matrix from sklearn.metrics.
# 
# 
# 4.Generate a classication report using the imbalanced_classification_report from imbalanced-learn.
# 
# 
# 5.For the Balanced Random Forest Classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score
# 
# Note: Use a random state of 1 for each algorithm to ensure consistency between tests 

# ### Balanced Random Forest Classifier

# In[16]:


# Resample the training data with the BalancedRandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
rf_model = BalancedRandomForestClassifier(n_estimators=100, random_state=1)
rf_model = rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


# In[17]:


# Calculated the balanced accuracy score
balanced_accuracy_score(y_test, y_pred)


# In[18]:


# Display the confusion matrix
confusion_matrix(y_test, y_pred)


# In[19]:


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))


# In[20]:


# List the features sorted in descending order by feature importance
sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)


# ### Easy Ensemble Classifier

# In[21]:


# Train the EasyEnsembleClassifier
from imblearn.ensemble import EasyEnsembleClassifier
eec = EasyEnsembleClassifier(random_state=1)
eec.fit(X_train, y_train)
y_pred = eec.predict(X_test)


# In[22]:


# Calculated the balanced accuracy score
balanced_accuracy_score(y_test, y_pred)


# In[23]:


# Display the confusion matrix
confusion_matrix(y_test, y_pred)


# In[24]:


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### Final Questions
# 
# #### Which model had the best balanced accuracy score? YOUR ANSWER HERE.
# #####  Looks like it's the Easy Classifer which is at .915
# 
# 
# #### Which model had the best recall score? YOUR ANSWER HERE.
# ##### High Risk edges things out at .93 to .90
# 
# 
# #### Which model had the best geometric mean score? YOUR ANSWER HERE.
# 
# ##### Tied at .92
# 
# #### What are the top three features? YOUR ANSWER HERE.
# 
# ##### If I read the question right, total_rec_prncp, total_payment, total_payment_inv

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




