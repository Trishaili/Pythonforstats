#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm


# In[4]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)


# In[5]:


boston_df.head()


# In[6]:


boston_df.info()


# In[7]:


boston_df.columns


# In[14]:


# Display a Boxplot for the "Median value of owner-occupied homes 
box = sns.boxplot(y = 'MEDV', data = boston_df)


# In[9]:


#The boxplot shown here appears to be skewed to the right.The median is close to the first quartile and the third quartile is further from the median.There are no outliers.


# In[12]:


#Provide a  bar plot for the Charles river variable
data = boston_df['CHAS']
labels = "Boxplot for Charles river variable"
plt.bar(labels,data)
plt.show


# In[13]:


# 1 if tract bounds variable;0 otherwise


# In[16]:


#boxplot for the MEDV variable vs the AGE variable. 
#(Discretize the age variable into three groups of 35 years and younger, between 35 and 70 years and 70 years and older)
boston_df.loc[(boston_df['AGE'] <= 35), 'age_group'] = '35 years and younger'
boston_df.loc[(boston_df['AGE'] > 35)&(boston_df['AGE'] < 70), 'age_group'] = 'between 35 and 70 years'
boston_df.loc[(boston_df['AGE'] >= 70), 'age_group'] = '70 years and older'
boxplot = sns.boxplot(x = 'age_group', y = 'MEDV', data = boston_df)



# In[17]:


#The boxplot appears to be relatively symmetric and has a few outliers on the lower end of the distribution.
#The median is around 70, 


# In[18]:


#scatter plot to show the relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town.
#NOX - nitric oxides concentration
#INDUS - proportion of non-retail business 
scatterplot = sns.scatterplot(x = 'INDUS', y = 'NOX', data = boston_df)


# In[19]:


#The Scatter Plot shows that there is high correlation between the Nitric Oxides concentration and Proportion of non-retail business acres per town. 


# In[21]:


# histogram for the pupil to teacher ratio variable
histogramplot =  sns.histplot(boston_df['PTRATIO'])


# In[22]:


#The histogram above appears to tbe skewed to the left which means it is a neagtive skewed distribution.


# In[23]:


#Is there a significant difference in median value of houses bounded by the Charles river or not? 
#(T-test for independent samples)
#H0:µ1=µ2("there is no difference in median value of houses bounded by the Charles river")
#H1:µ1≠µ2("there is a difference in median value of houses bounded by the Charles river")
scipy.stats.levene(boston_df['MEDV'], boston_df['CHAS'], center = 'mean')


# In[24]:


scipy.stats.ttest_ind(boston_df['MEDV'], boston_df['CHAS'])


# In[25]:


#Conclusion: Since the p-value is less than 0.05 we reject the null hypothesis as there is enough proof that there is a statistical difference in Median value of owner-occupied homes based on Charles river variable


# In[31]:


#Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)
#H0:µ1=µ2=µ3(the three population means are equal)
#H1:At least one of the means differ
scipy.stats.levene(boston_df[boston_df['age_group'] == '35 years and younger']['MEDV'],
                   boston_df[boston_df['age_group'] == 'between 35 and 70 years']['MEDV'], 
                   boston_df[boston_df['age_group'] == '70 years and older']['MEDV'], 
                   center='mean')


# In[33]:


thirtyfive_lower = boston_df[boston_df['age_group'] == '35 years and younger']['MEDV']
thirtyfive_seventy = boston_df[boston_df['age_group'] == 'between 35 and 70 years']['MEDV']
seventy_older = boston_df[boston_df['age_group'] == '70 years and older']['MEDV']


# In[34]:


#one way ANOVA
f_statistic, p_value = scipy.stats.f_oneway(thirtyfive_lower, thirtyfive_seventy, seventy_older)
print("F_Statistic: {0}, P-Value: {1}".format(f_statistic,p_value))


# In[35]:


#Conclusion: Since the p-value is less than alpha value 0.05, we reject the null hypothesis as there is enough proof 


# In[36]:


#Conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)
#H0:There is no relationship between NOX and INDUS.
#H1:There is a relationship between NOX and INDUS.
scipy.stats.pearsonr(boston_df['INDUS'], boston_df['NOX'])


# In[37]:


#Since the p-value < 0.05, we reject the Null hypothesis and conclude that there exists a relationship between NOX and INDUS.


# In[38]:


#Impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes? (Regression analysis)
#H0:β1=0 
#H1:β1≠0 
X = boston_df['DIS']
y = boston_df['MEDV']
X = sm.add_constant(X) 

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

model.summary()


# In[ ]:


#Since the p-value < 0.05, we reject the Null hypothesis and conclude that there exists a relationship between DIS and MEDV.

