#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn  as sns
from sklearn.linear_model import LinearRegression
sns.set()


# # Insert Raw Data

# In[7]:


raw_data=pd.read_csv('Cars.csv')
raw_data.head()


# # Preprocessing

# In[8]:


raw_data.describe(include='all')


# In[10]:


data=raw_data.drop(['Model'],axis=1) #312 unique models
data.describe(include='all')


# # Dealing with missing values

# In[12]:


data.isnull().sum()


# In[13]:


data_nv=data.dropna(axis=0) #drop null values from rows axis=0=rows


# # Exploring the Probability Distribution Functions

# In[15]:


sns.distplot(data_nv['Price']) #exponential  


# # Dealing with Outliers

# In[18]:


q=data_nv['Price'].quantile(0.99)
data_1=data_nv[data_nv['Price']<q] #remove the outliers at top 1%


# In[21]:


sns.distplot(data_1['Mileage'])


# In[19]:


q=data_1['Mileage'].quantile(0.99)
data_2=data_1[data_1['Mileage']<q]


# In[20]:


sns.distplot(data_2['Mileage'])


# In[23]:


sns.distplot(data_nv['EngineV'])


# In[24]:


#natural domain of Engine Volume is less than 6.5
data_3=data_2[data_2['EngineV']<6.5]
sns.distplot(data_3['EngineV'])


# In[25]:


sns.distplot(data_nv['Year'])


# In[26]:


q=data_3['Year'].quantile(0.01)
data_4=data_3[data_3['Year']>q]
sns.distplot(data_4['Year']) #drop at year lower 1%


# In[28]:


data_cleaned=data_4.reset_index(drop=True)


# In[29]:


data_cleaned.describe(include='all')


# In[30]:


#overall we deleted 250 problematic observations


# # Checking OLS assumptions

# In[33]:


f,(ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])                
ax2.set_title('Price and Engine Volume')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')


# In[34]:


sns.distplot(data_cleaned['Year'])
#use log plots when facing exponential relationships and linear regression isn't possible


# In[35]:


log_price=np.log(data_cleaned['Price'])
data_cleaned['log price']=log_price
data_cleaned.head()


# In[36]:


f,(ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log price'])                
ax2.set_title('Log Price and Engine Volume')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log price'])
ax3.set_title('Log Price and Mileage')
plt.show()
#all the assumptions are met


# In[38]:


data_cleaned=data_cleaned.drop(['Price'],axis=1) #no need of exponential Price


# # Check for Multicollinearity

# In[44]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables=data_cleaned[['Mileage','Year','EngineV']]
vif=pd.DataFrame()
vif['VIF']=[variance_inflation_factor(variables.values,i) for i in range (variables.shape[1])]
vif['Features']=variables.columns
vif #check for multicollinearity. If 1 then multicollinearity exists. Year is related to Mileage


# $VIF= 1/(1-R^2)$

# In[40]:


data_no_multicollinearity=data_cleaned.drop(['Year'] , axis=1)


# # Creating Dummy variables
# 

# In[45]:


# for n categorical features create n-1 dummies


# In[47]:


data_with_dummies=pd.get_dummies(data_no_multicollinearity,drop_first=True) #if first column isn't dropped,multicollinearity occurs
data_with_dummies.head()


# In[48]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variable =data_cleaned[['Brand','Body','Engine Type']]
vif_dummies=pd.DataFrame()
vif_dummies['VIF for Dummies']=[variance_inflation_factor(variables.values,i) for i in range (variables.shape[1])]
vif_dummies['Features']=variable.columns
vif_dummies 


# #  Rearrange a bit

# In[49]:


data_with_dummies.columns.values


# In[50]:


cols=['log price','Mileage', 'EngineV',  'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data_preprocessed=data_with_dummies[cols]


# # Linear Regression Model

# ## Declare the inputs and targets

# In[56]:


target=data_preprocessed['log price']
inputs=data_preprocessed.drop(['log price'],axis=1)


# ## scale the data

# In[57]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(inputs)


# In[58]:


inputs_scaled=scaler.transform(inputs)


# In[59]:


#scaling dummies has no effect on the predictive power but once scaled they lose all their dummy meaning


# ### Train Test Split
# 

# In[60]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs_scaled,targets,test_size=0.2,random_state=365)


# ### Create the regression

# In[61]:


reg=LinearRegression()
reg.fit(x_train,y_train) #create log linear regression model


# In[63]:


y_hat=reg.predict(x_train)
plt.scatter(y_train,y_hat)
plt.xlabel('Targets(y-train)',size=15)
plt.ylabel('Predictions(y-hat)',size=15)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[64]:


# we can draw a 45 degree line that shows the best possible match for targets and predictions. 
#Closer to the scatter plot line,better the model


# In[66]:


sns.distplot(y_train-y_hat)
plt.title('Residuals PDF',size=20)


# In[67]:


#overestimate the targets since negative tail is too big


# In[68]:


reg.score(x_train,y_train)


# ### finding weights and bias

# In[70]:


reg_summary=pd.DataFrame(inputs.columns.values,columns=['Features'])
reg_summary['Weights']=reg.coef_
reg_summary


# In[71]:


#a negative weight shows all -ve weights brands were cheaper than Audi since Audi=1 is the bench mark and +ve were more expensive


# # Testing

# In[72]:


y_hat_test=reg.predict(x_test)


# In[75]:



plt.scatter(y_test,y_hat_test,alpha=0.2)
plt.xlabel('Targets(y-test)',size=15)
plt.ylabel('Predictions(y-hat_test)',size=15)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[74]:


#for higher prices more concerntration of values near 45 degree line hence the model is very good for predicting higher prices
#but for lower prices the values are more scattered,hence we aren't quite getting the values right


# In[78]:


df_pf=pd.DataFrame(np.exp(y_hat_test),columns=['Predictions'] )
df_pf.head()


# In[79]:


df_pf['Target']=np.exp(y_test)
df_pf.head()


# In[80]:


y_test=y_test.reset_index(drop=True)
y_test


# In[81]:


df_pf['Target']=np.exp(y_test)
df_pf.head()


# In[82]:


df_pf['Residual']=df_pf['Target'] - df_pf['Predictions']


# In[83]:


df_pf['Difference %']=np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf.head()


# In[85]:


df_pf.describe()


# In[86]:


df_pf.sort_values(by=['Difference %'])


# In[87]:


#lower prices have much higher predictions than targets so we must be making a mistake in the predictions
#This could be due to -removal of the model of the car or damage to car


# In[ ]:




