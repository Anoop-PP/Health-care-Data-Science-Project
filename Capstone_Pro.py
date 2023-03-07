#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
#Robust Scaler
from sklearn.preprocessing import RobustScaler,StandardScaler
rs=RobustScaler()


from imblearn.over_sampling import SMOTE 
 
# Hold-out Method 
from sklearn.model_selection import train_test_split

# Grid Search CV
from sklearn.model_selection import GridSearchCV

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Tree Model
from sklearn.tree import DecisionTreeClassifier

# Support Vector Machine 
from sklearn.svm import SVC

# Ensemble Model
from lightgbm import LGBMClassifier 
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score


# In[2]:


x=pd.read_csv('health care diabetes.csv')
data=x


# In[3]:


x.head(10)


# In[8]:


features = x.iloc[:,[0,1,2,3,4,5,6,7]].values
label = x.iloc[:,8].values


# In[6]:


x.isnull().any()


# In[74]:


x.info()


# In[209]:


Positive = x[x['Outcome']==1]
Positive.head(10)


# In[210]:


x['Glucose'].value_counts().head(10)


# In[181]:


plt.hist(x['Glucose'],bins=20)


# In[78]:


x['BloodPressure'].value_counts().head(10)


# In[79]:


plt.hist(x['BloodPressure'],bins=20)


# In[80]:


x['SkinThickness'].value_counts().head(10)


# In[81]:


plt.hist(x['SkinThickness'],bins=20)


# In[82]:


x['Insulin'].value_counts().head(10)


# In[83]:


plt.hist(x['Insulin'])


# In[84]:


x['BMI'].value_counts().head(10)


# In[85]:


plt.hist(x['BMI'])


# In[211]:


x.describe().transpose()


# In[212]:


P=round(Positive.describe().transpose(),2)
P


# In[213]:


Negative = x[x['Outcome']==0]
N=round(Negative.describe().transpose(),2)
N


# # We can see the presence of outliers for Insulin and Pregnancies

# In[214]:


plt.hist(Positive['Glucose'],bins=20,edgecolor='black')


# In[90]:


Positive['Glucose'].value_counts().head(10)


# In[91]:


plt.hist(Positive['BloodPressure'],bins=20,edgecolor='black')


# In[92]:


Positive['BloodPressure'].value_counts().head(10)


# In[93]:


plt.hist(Positive['SkinThickness'],bins=20,edgecolor='black')


# In[94]:


Positive['SkinThickness'].value_counts().head(10)


# In[95]:


plt.hist(Positive['Insulin'],bins=20,edgecolor='black')


# In[96]:


Positive['Insulin'].value_counts().head(10)


# In[97]:


plt.hist(Positive['BMI'],bins=20,edgecolor='black')


# In[98]:


Positive['BMI'].value_counts().head(10)


# In[99]:


x_box=x.drop('Outcome',axis=1)


# In[215]:


fig,ax=plt.subplots(nrows=2,ncols=4,figsize=(20,10))
ax=ax.flatten()
index=0
for i in x_box.columns:
    sns.boxplot(y=i,data=x_box,ax=ax[index],color='green')
    index +=1
plt.tight_layout(pad=0.4)    


# 

# In[216]:


corr=x.corr()
sns.heatmap(corr,fmt='.1f',linewidth=0.2,linecolor='black',annot=True,cmap='PuBuGn')


# In[217]:


sns.pairplot(x,hue='Outcome')


# In[ ]:


# Above plot shows the distribution of dataset in scatter plot and KDE for diff combination


# In[105]:


sns.kdeplot(Negative.Pregnancies,shade=True,color='r')
sns.kdeplot(Positive.Pregnancies,shade=True,color='g')
plt.title('Analysis for Pregnancies for Diabetic and Non Diabetic')


# In[ ]:


# Pregnancies distribution for (red line)negative is right skewed and for positive(green line)the distribution is abnormal


# In[106]:


sns.kdeplot(Negative.Glucose,shade=True,color='r')
sns.kdeplot(Positive.Glucose,shade=True,color='g')
plt.title('Analysis for Glucose for Diabetic and Non Diabetic')


# In[ ]:


# In this KDE plot, the data for negative is distributed like normal/gaussian distribution, with sharp peakness
# in positive case the distribution is abnormal,with some outliers


# In[107]:


sns.kdeplot(Negative.BloodPressure,shade=True,color='r')
sns.kdeplot(Positive.BloodPressure,shade=True,color='g')
plt.title('Analysis for BP for Diabetic and Non Diabetic')


# In[ ]:


# The above shows that the distribution is almost similar for both positive and negative


# In[108]:


sns.kdeplot(Negative.SkinThickness,shade=True,color='r')
sns.kdeplot(Positive.SkinThickness,shade=True,color='g')
plt.title('Analysis for SkinThickness for Diabetic and Non Diabetic')


# In[ ]:


# Here both curve has mode equal to 0. 
# The tail part of Negative extended little more, which indicates the presence of outliers


# In[109]:


sns.kdeplot(Negative.Insulin,shade=True,color='r')
sns.kdeplot(Positive.Insulin,shade=True,color='g')
plt.title('Analysis for Insulin for Diabetic and Non Diabetic')


# In[ ]:


# Both plot are right skewed, positive plot is little extended and shows the outliers


# In[110]:


sns.kdeplot(Negative.BMI,shade=True,color='r')
sns.kdeplot(Positive.BMI,shade=True,color='g')
plt.title('Analysis for BMI for Diabetic and Non Diabetic')


# In[ ]:


# Positive curve is a normal distribution and indicates that abnormal in BMI will cause Diabetes


# In[111]:


sns.kdeplot(Negative.Age,shade=True,color='r')
sns.kdeplot(Positive.Age,shade=True,color='g')
plt.title('Analysis for Age for Diabetic and Non Diabetic')


# In[ ]:


# Negative is right skewed and mode value of 22


# In[112]:


sns.relplot(x='Glucose',y='BMI',data=x,hue='Outcome')


# In[ ]:


# The above scatter plot indicates, higher glucose value and abnormal BMI value have high chance of getting diabetic


# In[113]:


sns.relplot(x='Age',y='BloodPressure',data=x,hue='Outcome')


# In[ ]:


#here indicates, people with high BP and aged have higher chance to be diabetic


# In[114]:


sns.relplot(x='Glucose',y='BloodPressure',data=x,hue='Outcome')


# In[ ]:


#This plot indicates, people with high glucose and BP have high chances of being Diabetic


# In[ ]:





# # Binning the columns 

# In[218]:


x['Age_x']=pd.cut(x=x['Age'],bins=[20,30,50,100],labels=['Young Aged','Middle Aged','old Aged'])
x.head(10)


# # Treating BMI

# In[219]:


plt.hist(x['BMI'])


# In[189]:


Negative=(x['Outcome']==0)
Positive=(x['Outcome']==1)
Avg_Negative=x.loc[Negative,'BMI'].median()
Avg_Positive=x.loc[Positive,'BMI'].median()
x.loc[x['BMI']==0 & Negative, 'BMI']=Avg_Negative
x.loc[x['BMI']==0 & Positive, 'BMI']=Avg_Positive


# In[190]:


x.BMI.hist(bins=20)


# In[220]:



sns.boxplot(x.BMI)


# In[ ]:


#We need to convert it into categorical column, since the presence of outliers


# In[221]:


x['BMI_bin']=pd.cut(x['BMI'],bins=[18,25,35,80],labels=['Normal','Overweight','obese'])
x.head(10)


# In[201]:


BMI_bin = x.BMI_bin
x.insert(0,'BMI_binned',BMI_bin)


# In[203]:


x = x.drop(['BMI_bin','BMI'],axis=1)
x.head()


# # Treating Insulin

# In[193]:


x.Insulin.hist(bins=20,grid=False)


# In[194]:


Negative=(x['Outcome']==0)
Positive=(x['Outcome']==1)
Avg_Negative=x.loc[Negative,'Insulin'].median()
Avg_Positive=x.loc[Positive,'Insulin'].median()
x.loc[x['Insulin']==0 & Negative,'Insulin']=Avg_Negative
x.loc[x['Insulin']==0 & Positive,'Insulin']=Avg_Positive


# In[195]:


x.Insulin.hist(bins=20,grid=False)


# In[ ]:


#  The above plot shows, even after replacing 0 with median there is no proper change.


# # Insulin Log() Transformation

# In[196]:


x.Insulin=np.log(x.Insulin)


# In[197]:


x.Insulin.hist(bins=20,grid=False)


# In[125]:


sns.boxplot(x.Insulin)


# In[ ]:


#In above plot, after applying the Log transformation median and IQR has been changed.
# Since the presence of Outliers , it is to be scaled using Robust Scaler


# In[126]:


RS=RobustScaler(with_centering=True,with_scaling=True,quantile_range=(25.0,75.0),copy=True)


# In[127]:


x['Insulin']=RS.fit_transform(x['Insulin'].values.reshape(-1,1))


# In[128]:


x.Insulin.hist(bins=20,grid=False)


# In[129]:


x.BloodPressure.hist(bins=20,grid=False)


# In[131]:


Negative=(x['Outcome']==0)
Positive=(x['Outcome']==1)
Avg_Negative=x.loc[Negative,'BloodPressure'].median()
Avg_Positive=x.loc[Positive,'BloodPressure'].median()
x.loc[x['BloodPressure']==0 & Negative,'BloodPressure']=Avg_Negative
x.loc[x['BloodPressure']==0 & Positive,'BloodPressure']=Avg_Positive


# In[132]:


x.BloodPressure.hist(bins=20,grid=False)


# In[133]:


from sklearn.preprocessing import StandardScaler


# In[134]:


S_Scale=StandardScaler(copy=True,with_mean=True,with_std=True)


# In[135]:


x['BloodPressure']=rs.fit_transform(x['BloodPressure'].values.reshape(-1,1))


# In[136]:


x.BloodPressure.hist(bins=20,grid=False)


# In[137]:


x.Glucose.hist(bins=20,grid=False)


# In[138]:


Negative=(x['Outcome']==0)
Positive=(x['Outcome']==1)
Avg_Negative=x.loc[Negative,'Glucose'].median()
Avg_Positive=x.loc[Positive,'Glucose'].median()
x.loc[x['Glucose']==0 & Negative,'Glucose']=Avg_Negative
x.loc[x['Glucose']==0 & Positive,'Glucose']=Avg_Positive


# In[139]:


x.Glucose.hist(bins=20,grid=False)


# In[140]:


x['Glucose']=rs.fit_transform(x['Glucose'].values.reshape(-1,1))


# In[141]:


x.Glucose.hist(bins=20,grid=False)


# In[142]:


x.SkinThickness.hist(bins=20,grid=False)


# In[143]:


Negative=(x['Outcome']==0)
Positive=(x['Outcome']==1)
Avg_Negative=x.loc[Negative,'SkinThickness'].median()
Avg_Positive=x.loc[Positive,'SkinThickness'].median()
x.loc[x['SkinThickness']==0 & Negative,'SkinThickness']=Avg_Negative
x.loc[x['SkinThickness']==0 & Positive,'SkinThickness']=Avg_Positive


# In[144]:


x.SkinThickness.hist(bins=20,grid=False)


# In[145]:


sns.boxplot(x.SkinThickness)


# In[ ]:


# In above plot, presence of outliers are seen, so it is treated with Robust scaler


# In[146]:


x['SkinThickness']=rs.fit_transform(x['SkinThickness'].values.reshape(-1,1))


# In[147]:


x.SkinThickness.hist(bins=20,grid=False)


# In[148]:


x.head(10)


# In[149]:


x=x.drop('Age',axis=1)
x.head()


# In[150]:


Age_Bin=x.Age_x
x.insert(0,'Aged_Bin',Age_Bin)
x=x.drop('Age_x',axis=1)
x.head()


# In[151]:


x.Aged_Bin=x.Aged_Bin.replace(to_replace=['Young Aged','Middle Aged','old Aged'],value=[0,1,2],inplace=False)
x.head()


# In[152]:


x.BMI_bin=x.BMI_bin.replace(to_replace=['Normal','Overweight','obese'],value=[0,1,2],inplace=False)
x.head()


# In[153]:


x.Outcome.value_counts()


# In[ ]:


#Here the data is imbalanced so it is oversampled


# # Oversampling

# In[154]:


from imblearn.over_sampling import SMOTE
smt=SMOTE(sampling_strategy='auto',random_state=9,n_jobs=1)
x1=x.drop(['Outcome'],axis=1)
y1=x.Outcome
x1,y1=smt.fit_sample(x1,y1)


# In[ ]:


#Hold Out method


# In[155]:


x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.2,random_state=9,stratify=y1)


# In[156]:


print('Shape of x_train',x_train.shape)
print('\nShape of y_train',y_train.shape)
print('\nShape of x_test',x_test.shape)
print('\nShape of y_test',y_test.shape)


# # Logistic Regression

# In[158]:


lr=LogisticRegression(random_state=100,n_jobs=-1,penalty='l2',solver='liblinear')
lr.fit(x_train,y_train)
y_pred_lr=lr.predict(x_test)


# In[171]:


print('Test Accuracy :-',accuracy_score(y_pred_lr,y_test))


# In[ ]:





# # Decision Tree Classifier with Oversampling

# In[160]:


dtc=DecisionTreeClassifier(criterion='entropy',splitter='best',random_state=9)


# In[161]:


dtc.fit(x_train,y_train)
y_pred_dtc=dtc.predict(x_test)


# In[162]:


print('Test Accuracy :-',accuracy_score(y_pred_dtc,y_test))


# # Random Forest Classifier with Oversampling

# In[163]:


rfc=RandomForestClassifier(max_depth=2,random_state=0,n_jobs=-1)
rfc.fit(x_train,y_train)
y_pred_rfc=rfc.predict(x_test)


# In[164]:


print('Test Accuracy :-',accuracy_score(y_pred_rfc,y_test))


# In[165]:


# Without Oversampling
xs=x.drop('Outcome',axis=1)
ys=x.Outcome
x_train,x_test,y_train,y_test=train_test_split(xs,ys,test_size=0.2,random_state=9)


# In[166]:


print('Shape of x_train',x_train.shape)
print('\nShape of y_train',y_train.shape)
print('\nShape of x_test',x_test.shape)
print('\nShape of y_test',y_test.shape)


# In[167]:


#Decision Tree Classifier
dtc1=DecisionTreeClassifier(criterion='entropy',splitter='best',random_state=9)
dtc1.fit(x_train,y_train)
y_pred_dtc1=dtc1.predict(x_test)
print('Test Accuracy :-',accuracy_score(y_pred_dtc1,y_test))


# In[168]:


#Random Forest Classifier 
rfc1=RandomForestClassifier(max_depth=2,random_state=0,n_jobs=-1)
rfc1.fit(x_train,y_train)
y_pred_rfc1=rfc1.predict(x_test)
print('Test Accuracy :-',accuracy_score(y_pred_rfc1,y_test))


# # ROC AUC curve for Random Forest Classifier

# In[169]:


fpr, tpr, thershold = roc_curve(y_test, rfc1.predict_proba(x_test)[:,1])
rfc_roc = roc_auc_score(y_pred_rfc1,y_test)
plt.figure()
plt.subplots(figsize=(15,10))
plt.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)'%rfc_roc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0,1.0])
plt.ylim([0,1.01])
plt.xlabel('False Positive Rate (1-specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.title('Receiver operating characteristic for Decision Tree Classifier ')
plt.legend(loc ="lower right")
plt.show()


# # SVM  and GridSearchCV

# In[170]:


C=np.logspace(-2,2,5)
gamma=np.logspace(-5,5,5)
kernel=['linear', 'rbf', 'sigmoid']
param_grid = dict(C=C,gamma=gamma,kernel=kernel)


# In[171]:


grid = GridSearchCV(SVC(),param_grid=param_grid,n_jobs=-1)


# In[172]:


grid.fit(x_train,y_train)
y_pred_grid = grid.predict(x_test)
print('Grid Search best parameter for SVC are : ',grid.best_params_)
print()
print('SVC predicted accuracy score is :---- ', accuracy_score (y_pred_grid,y_test))


# # Light GBM Classifier

# In[173]:


lgbm = LGBMClassifier(boosting_type='goss',  
                      n_jobs=-1,
                      objective='binary',
                      random_state=9,
                      importance_type='split'
                      )


# In[174]:


lgbm.fit(x_train, y_train)
y_pred_lgbm = lgbm.predict(x_test)
print('lgbm predicted accuracy score is : ', accuracy_score (y_pred_lgbm,y_test))


# In[227]:


features = data.iloc[:,[0,1,2,3,4,5,6,7]].values
label = data.iloc[:,8].values


# In[228]:



from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = rfc1.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = rfc1.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')

