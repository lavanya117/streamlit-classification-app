import streamlit as st
#!/usr/bin/env python
# coding: utf-8

st.markdown('# Exploratory Data Analysis')

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[3]:


df=pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')
df.head()


# In[4]:


df['Gender'].unique()


# In[5]:


x=df['Loan_ID'].duplicated().sum()
st.write(f"There are {x} duplicate Loan IDs")


# In[6]:


df['Education'].unique()


# In[7]:


df.Property_Area.unique()	


st.markdown('**Null values in the data**')

# In[9]:


for label in df.columns:
    st.write(f"There are {df[label].isnull().sum()} null values in {label}.")


# In[10]:


df.Loan_Status.unique()


# In[11]:


x=df.isnull().sum().sum()
st.write(f"There are {x} null values in the dataset")


st.markdown('**Dropping a column (Loan ID)**')

# In[13]:


df=df.drop(['Loan_ID'],axis=1)


st.markdown('**Filling null values**')

# In[15]:


df['Gender']=df['Gender'].fillna('Female');
df['Married']=df['Married'].fillna(df['Married'].mode()[0]);
df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0]);
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0]);
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mode()[0]);
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0]);
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].median());


# In[16]:


df.head()


# In[17]:


df[df['Loan_Status']=='Y'].value_counts().sum()


# In[18]:


df[df['Loan_Status']=='N'].value_counts().sum()


st.markdown('**Pairplot**')

# In[20]:


sns.pairplot(df)
st.pyplot(plt)


# In[21]:


st.write(f"There are {df.isnull().sum().sum()} null values now.")


st.markdown('**Data Visualisation using Matplotlib and Seaborn**')

# In[23]:


heat=df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]
heat=heat.corr()
sns.heatmap(heat,annot=True)
st.pyplot(plt)

st.markdown('**How many Loans were approved?**')

# In[25]:


st.write(f"{df[df['Loan_Status']=='Y'].value_counts().sum()} loans were approved")


st.markdown('**How many Loans were rejected?**')

# In[27]:


st.write(f"{df[df['Loan_Status']=='N'].value_counts().sum()} loans were rejected")


st.markdown('**How many married people applied for the loans?**')

# In[29]:


st.write(f"{df[df['Married']=='Yes'].value_counts().sum()} married people got their loan approved")


st.markdown("**How many married people's loan got accepted?**")

# In[31]:


x=df[(df['Married'] == 'Yes') & (df['Loan_Status'] == 'Y')].shape[0]
st.write(f"{x} married people got their loan rejected")


st.markdown("**How many unmarried people's loan got accepted?**")

# In[33]:


y=df[(df['Married']=='No') & (df['Loan_Status'] == 'Y')].shape[0]
st.write(f"{y} unmarried people got their loan approved")


# In[34]:


value=[x,y]
label=['Married','Unmarried']
fig, ax = plt.subplots()
ax.pie(value, labels=label, autopct='%0.1f%%')
ax.set_title("People with Approved Loans")
st.pyplot(fig)

st.markdown("**How many self-employed people's loan got accepted?**")

# In[36]:


st.write(f"{df[(df['Self_Employed']=='Yes') & (df['Loan_Status']=='Y')].shape[0]} self employed people got their loan approved")


st.markdown("**How many people with dependents had their loan approved?**")

# In[38]:


st.write(f"{df[(df['Dependents']!=0) & (df['Loan_Status']=='Y')].shape[0]} dependents got their loan approved")


st.markdown("**Is there any difference in the loan status because of gender?**")

# In[40]:


T=df[df['Loan_Status']=='Y'].groupby('Gender').size()
st.write(T) 


st.markdown("**No of males present in the data**")

# In[42]:


total_male=df[df['Gender']=='Male'].shape[0]
st.write(f"There are total {total_male} males present in the data")  


st.markdown("**No of females present in the data**")
# 

# In[44]:


total_female=df[df['Gender']=='Female'].shape[0]
st.write(f"There are total {total_female} females present in the data")


st.markdown('**Percentage**')
# 
st.markdown("**What percentage of females got their loan approved?**")
st.markdown('%percentage=(83/125)*100=66.40%')
# 
st.markdown("**What percentage of males got their loan approved?**")
st.markdown('%percentage=(339/489)*100=69.32%')
# 
# 

st.markdown("**How many female and male got there loan approved and how many did not?**")

# In[47]:


sns.histplot(data=df, x='Loan_Status', hue='Gender')
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.xticks([0,1],['Rejected','Approved'])
plt.title("Loan status of both genders")
st.pyplot(plt)


st.markdown("**What is the range of loan amount of approved loan?**")

# In[49]:


plt.figure(figsize=(10,10))
sns.boxplot(data=df,x='Loan_Status',y='LoanAmount',hue='Loan_Status')
st.pyplot(plt)


st.markdown("**Adding a new column in the dataset named as total income**")
# 
st.markdown('Total income=Applicant Income + Co-applicant Income')

# In[51]:


new_col=df['ApplicantIncome']+df['CoapplicantIncome']
df.insert(len(df.columns)-1,'Total_income',new_col.values)
df.head()


# In[52]:


plt.figure(figsize=(10,10))
sns.boxplot(data=df,x='Loan_Status',y='Total_income',hue='Loan_Status')
st.pyplot(plt)


st.markdown('**Mathematical information about the data**')

# In[54]:


df.describe()


st.markdown('**Loan amount vs Total income**')

# In[189]:


plt.figure(figsize=(10,10))
sns.scatterplot(data=df, x='Total_income', y='LoanAmount', hue='Loan_Status',palette=['blue','orange'])
plt.legend(title='Loan_Status',labels=['Rejected','Approved'])
st.pyplot(plt)


st.markdown('## Feature Engineering')

# In[58]:


from sklearn.preprocessing import OneHotEncoder
inn=['Education','Property_Area','Gender']
ls=df['Loan_Status']
df=df.drop(['Loan_Status'],axis=1)
ohe=OneHotEncoder(sparse_output=False).set_output(transform='pandas')
for label in inn:
    ohetrans=ohe.fit_transform(df[[label]])
    df=df.drop([label],axis=1)
    df=pd.concat([df,ohetrans],axis=1)
df=pd.concat([df,ls],axis=1)    


# In[59]:


df['Loan_Status']=(df['Loan_Status']=='Y').astype(int)
df['Self_Employed']=(df['Self_Employed']=='Yes').astype(int)
df['Married']=(df['Married']=='Yes').astype(int)
df['Dependents']=df['Dependents'].replace({'3+':3}).astype(int)
df.head()


st.markdown('**Splitting the data**')

# In[61]:


X=df.iloc[:,:16].values
Y=df.iloc[:,-1].values


# In[62]:


from imblearn.over_sampling import SMOTE
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1,test_size=0.2)
sm=SMOTE(random_state=42)
X_train,Y_train=sm.fit_resample(X_train,Y_train)


st.markdown('**Scaling the data**')

# In[64]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[65]:


pd.DataFrame(X_train)


st.markdown('# Logistic Regression')

# In[67]:


from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()


# In[68]:


from sklearn.model_selection import GridSearchCV
parameter = {
    'penalty': ['l1', 'l2', 'elasticnet', None], # Corrected quotes and format
    'solver': ['liblinear','saga'],
    'multi_class': ['auto', 'ovr']
}
cl=GridSearchCV(reg,param_grid=parameter,scoring='accuracy',cv=5) #cv means cross validation
cl.fit(X_train,Y_train)


st.markdown('**Best parameters**')

# In[69]:


st.write(cl.best_params_)


st.markdown('**Best score**')

# In[70]:


st.write(cl.best_score_)


# In[71]:


Y_hat=cl.predict(X_test)


st.markdown('**Metrics**')

# In[72]:


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,f1_score


# In[172]:


score=accuracy_score(Y_test,Y_hat)
st.write(f"{score} is the accuracy score")


st.markdown('**Classification Report**')

# In[74]:


st.write(classification_report(Y_test,Y_hat))


st.markdown('**Confusion Matrix**')

# In[75]:


C=confusion_matrix(Y_test,Y_hat)


# In[76]:


plt.figure(figsize=(6, 5))
sns.heatmap(C, annot=True, xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(plt)


st.markdown('**F1 score**')

# In[77]:


st.write(f1_score(Y_test,Y_hat))


st.markdown('# Support Vector Classifier')

# In[79]:


from sklearn.svm import SVC
support=SVC()
parameter = {
    'kernel':['linear','poly','rbf','sigmoid'],
    'gamma':['scale', 'auto'],
    'decision_function_shape':['ovo','ovr']
}
sv=GridSearchCV(support,param_grid=parameter,scoring='accuracy',cv=5) 
sv.fit(X_train,Y_train)


st.markdown('**Best parameters and Best Accuracy Score**')

# In[80]:


st.write("Best Parameters:", sv.best_params_)
st.write("Best Accuracy Score:", sv.best_score_)


# In[81]:


Y_hat=sv.predict(X_test)


st.markdown('**Accuracy Score**')

# In[82]:


score=accuracy_score(Y_test,Y_hat)
st.write(score)


st.markdown('**Classification Report**')

# In[83]:


st.write(classification_report(Y_test,Y_hat))


st.markdown('**Confusion Matrix**')

# In[84]:


C=confusion_matrix(Y_test,Y_hat)


# In[85]:


plt.figure(figsize=(6, 5))
sns.heatmap(C, annot=True, xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(plt)


st.markdown('**F1 Score**')

# In[86]:


st.write(f1_score(Y_test,Y_hat))


st.markdown('# Decision Trees')

# In[88]:


from sklearn import tree


# In[89]:


dt=tree.DecisionTreeClassifier()


# In[90]:


parameter={
    'criterion':['gini', 'entropy', 'log_loss'],
    'splitter':['best', 'random']
}
de=GridSearchCV(dt,param_grid=parameter,scoring='accuracy',cv=5) 
de.fit(X_train,Y_train)


st.markdown('**Best parameters and Best Accuracy Score**')

# In[91]:


st.write("Best Parameters:", de.best_params_)
st.write("Best Accuracy Score:", de.best_score_)


# In[92]:


Y_hat=de.predict(X_test)
score=accuracy_score(Y_test,Y_hat)
st.write(score)


st.markdown('**Classification Report**')

# In[93]:


st.write(classification_report(Y_test,Y_hat))


st.markdown('**Confusion Matrix**')

# In[94]:


C=confusion_matrix(Y_test,Y_hat)


# In[95]:


plt.figure(figsize=(6, 5))
sns.heatmap(C, annot=True, xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(plt)


st.markdown('**F1 Score**')

# In[96]:


st.write(f1_score(Y_test,Y_hat))


# In[ ]:




