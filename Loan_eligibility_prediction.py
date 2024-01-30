#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', ' inline')


# In[3]:


dataset = pd.read_csv("loan-train.csv")


# In[4]:


dataset.head()


# In[6]:


dataset.shape


# In[7]:


dataset.info()


# In[8]:


dataset.describe()


# In[10]:


pd.crosstab(dataset['Credit_History'], dataset['Loan_Status'], margins=True)


# In[12]:


dataset.boxplot(column='ApplicantIncome')


# In[13]:


dataset['ApplicantIncome'].hist(bins=20)


# In[14]:


dataset['CoapplicantIncome'].hist(bins=20)


# In[15]:


dataset.boxplot(column='ApplicantIncome', by= 'Education')


# In[16]:


dataset.boxplot(column='LoanAmount')


# In[17]:


dataset['LoanAmount'].hist(bins=20)


# In[18]:


dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)


# In[19]:


dataset.isnull().sum()


# In[38]:


dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)


# In[39]:


dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)


# In[40]:


dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)


# In[41]:


dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)


# In[42]:


dataset.LoanAmount = dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log = dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())


# In[43]:


dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)


# In[44]:


dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)


# In[45]:


dataset.isnull().sum()


# In[46]:


dataset['TotalIncome']= dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome_log']= np.log(dataset['TotalIncome'])


# In[47]:


dataset['TotalIncome_log'].hist(bins=20)


# In[48]:


dataset.head()


# In[50]:


X= dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y= dataset.iloc[:,12].values


# In[51]:


X


# In[52]:


y


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0) 


# In[54]:


print(X_train)


# In[56]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()


# In[57]:


for i in range(0,5):
    X_train[:,i]= labelencoder_X.fit_transform(X_train[:,i])


# In[58]:


X_train[:,7]= labelencoder_X.fit_transform(X_train[:,7])


# In[59]:


X_train


# In[60]:


labelencoder_y=LabelEncoder()
y_train=labelencoder_y.fit_transform(y_train)


# In[61]:


y_train


# In[62]:


for i in range(0,5):
    X_test[:,i]= labelencoder_X.fit_transform(X_test[:,i]) 


# In[63]:


X_test[:,7]= labelencoder_X.fit_transform(X_test[:,7])


# In[64]:


labelencoder_y=LabelEncoder()
y_test=labelencoder_y.fit_transform(y_test)


# In[65]:


X_test


# In[66]:


y_test


# In[67]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)


# In[69]:


from sklearn.tree import DecisionTreeClassifier
DTClassifier=DecisionTreeClassifier(criterion='entropy', random_state=0)
DTClassifier.fit(X_train,y_train)


# In[70]:


y_pred=DTClassifier.predict(X_test)
y_pred


# In[71]:


from sklearn import metrics
print('The accuracy of decision tree is:', metrics.accuracy_score(y_pred,y_test))


# In[73]:


from sklearn.naive_bayes import GaussianNB
NBClassifier = GaussianNB()
NBClassifier.fit(X_train,y_train)


# In[74]:


y_pred= NBClassifier.predict(X_test)


# In[75]:


y_pred


# In[76]:


print('the accuracy of Naive bayes is:', metrics.accuracy_score(y_pred,y_test))


# In[77]:


testdata=pd.read_csv("loan-test.csv")


# In[78]:


testdata.head()


# In[79]:


testdata.info()


# In[80]:


testdata.isnull().sum()


# In[81]:


testdata['Gender'].fillna(testdata['Gender'].mode()[0],inplace=True)
testdata['Dependents'].fillna(testdata['Dependents'].mode()[0],inplace=True)
testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0],inplace=True)
testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0],inplace=True)
testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0],inplace=True)


# In[82]:


testdata.isnull().sum()


# In[83]:


testdata.boxplot(column='LoanAmount')


# In[84]:


testdata.boxplot(column='ApplicantIncome')


# In[85]:


testdata.LoanAmount= testdata.LoanAmount.fillna(testdata.LoanAmount.mean())


# In[86]:


testdata['LoanAmount_log']=np.log(testdata['LoanAmount'])


# In[87]:


testdata.isnull().sum()


# In[88]:


testdata['TotalIncome']=testdata['ApplicantIncome']+testdata['CoapplicantIncome']
testdata['TotalIncome_log']=np.log(testdata['TotalIncome'])


# In[89]:


testdata.head()


# In[90]:


test=testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[91]:


for i in range(0,5):
    test[:,i]=labelencoder_X.fit_transform(test[:,i])


# In[92]:


test[:,7]=labelencoder_X.fit_transform(test[:,7])


# In[93]:


test


# In[94]:


test= ss.fit_transform(test)


# In[95]:


pred= NBClassifier.predict(test)


# In[96]:


pred


# In[ ]:




