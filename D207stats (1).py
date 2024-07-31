#!/usr/bin/env python
# coding: utf-8

# # ************************************************************************************************************

# # Dr. Kesselly Kamara
# # D207 - Exploratory Data Analysis/Descriptive Analytics
# #Considered the simplest is used on historical data to discover trends and relationships in the data.
# # 1. Select one of the following methods to perform analysis (Chi-square, ANOVA, t-test)
# # 2. Perform statistical testing by using a hypothesis.
# # 3 Hypothesis testing: is a formal process for applying statistics to examine theories about the world.
# # 4. Variable: a container that holds values (categorical/numerical data). 
# # 5. Categorical: qualitative data
# # 6. Numerical: quantitative data (continuous or discrete data)
# # 8. Continuous: measurable numerical data
# # 9. Discrete: countable numerical data

# # ********************************************************************************************************

# # Steps:
# # 1. Define your practical Theory (e.g. gender is related to smoking) 
# # 2. Determine the method: (e.g. Chi-square)
# # 3 State your hypothesis: (null hypothesis -Ho> gender is not related to smoking, Ha> gender is related to smoking
# # 4. The null hypothesis is rejected if the p-value < 0.05  
# # 5. Collect data (Gender and smoking and Day and Smoking)
# # 6. Hypothesis (test to reject or accept)
# # 7. Report finding: practical conclusion

# # *****************************************************************************************************************

# # Dr. Kesselly Kamara

# # Python version 

# In[1]:


from platform import python_version
print(python_version())


# # ********************************************************************************************************************

# # Install and import appropriate libraries.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sn
import researchpy as rp

# chi-square test
from scipy.stats import chi2_contingency

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# # **********************************************************************************************************************

# # Collect Data

# In[3]:


data=sn.load_dataset('tips')


# In[4]:


df = data.rename(columns={'sex':'gender', 'smoker':'smoking'}) 


# In[5]:


df.head()


# # *************************************************************************************************************

# In[6]:


# Calculate the cross_tab to determine frequencies
cross_tab=pd.crosstab(index=df['day'],columns=df['smoking'])
cross_tab


# In[7]:


chi_result=chi2_contingency(cross_tab)


# In[8]:


def is_related(x,y):
    ct=pd.crosstab(index=df[x],columns=df[y])
    chi_result=chi2_contingency(ct)
    p, x=chi_result[1], "related" if chi_result[1] < 0.05 else "is not related"
    return p,x


# In[14]:


is_related('day', 'smoking') # Cannot reject the null hypothesis. Practical conclusion day has no impact on smoking.


# # *******************************************************************************************************************

# # ANOVA - Analysis of variance

# In[11]:


da=sn.load_dataset('diamonds')
da.head()


# # ANOVA or Analysis of Variance test to see if there are differences between two groups. 
# #Ho hypothesis is there is no association among the variables. 

# # One-way ANOVA has one independent categorical variables 

# In[33]:


df=da[['price','cut', 'clarity', 'depth', 'table' ]]
df.head()


# In[29]:


one_way=ols('tip~day',data=df).fit()
one_ano=sm.stats.anova_lm(one_way, type=2)
one_ano


# # Two-way ANOVA has two independent categorical variables
# #https://www.scribbr.com/statistics/one-way-anova/

# In[32]:


anov2=ols('tip~gender+day', data=df).fit()
t_way=sm.stats.anova_lm(anov2, type=2)
t_way


# # *******************************************************************

# In[16]:


def plot_hist(col_name, num_bins, do_rotate=False):
     plt.hist(data[col_name], bins=num_bins)
     plt.xlabel(col_name)
     plt.ylabel('Frequency')
     plt.title(f'Histogram of {col_name}')
     if do_rotate:
         plt.xticks(rotation=90)
     plt.show()
    
#function to describe column
def print_desc(col_name):
 print(data[col_name].describe())


# # The t-test is used to test the evaluate a population means

# # One Sample t-test 
# #One-sample t-test is used to evaluate a population means using a single sample.
# 
# #Problem: Five diabetes patients were randomly selected from a treatment. The doctor wants patients to have a glucose score of 110
# 
# #The Five patients glucose are 80, 90, 135, 140, 150. Can the doctor be 95% confident that the glucose average is 110
# 
# #Ho =  The group means is 110
# 
# #Ha = the means is not 110

# In[17]:


from scipy import stats as st
glucose = [80, 90, 135, 140, 150]
one_sample=st.ttest_1samp(glucose, 110)
one_sample


# # ------------------------------------------------------------------------------------------------------------------

# # Two Sample t-test 
# #two-sample t-test is used to evaluate a population means using more than one sample.

# In[43]:


two_sample=rp.ttest(group1= data['tip'][data['sex'] == 'Male'], group1_name= "Male",
         group2= data['tip'][data['sex'] == 'Female'], group2_name= "Female")
two_sample

