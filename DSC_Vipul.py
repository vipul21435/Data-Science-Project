#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/rites/Downloads/DSC_project.csv")
df


# In[2]:


df.describe()


# In[3]:


df.isnull()


# In[4]:


df.isnull().sum()


# In[5]:


df[df['label']==0].shape, df[df['label']==1].shape


# In[6]:


df.drop('Robots',axis=1,inplace = True)


# In[7]:


df.isnull().sum()


# In[8]:


df.dropna(axis=0,thresh=56*0.90,inplace=True)


# In[9]:


df.reset_index(drop=True,inplace=True)


# In[10]:


df.dtypes


# In[11]:


for index in range(len(df)):
    try:
        float(df["TLD"][index])
        df["TLD"][index] = pd.NA
    except ValueError:
        pass


# In[12]:


for index in range(len(df)):
    try:
        float(df["Title"][index])
        df["Title"][index] = pd.NA
    except ValueError:
        pass


# In[13]:


df.isnull().sum()


# In[14]:


df.dropna(subset='Title',inplace=True)


# In[15]:


df.reset_index(drop=True,inplace=True)


# In[16]:


df.isnull().sum()


# In[17]:


df.dropna(subset='URLLength',inplace=True)


# In[18]:


df.reset_index(drop=True,inplace=True)


# In[19]:


df.dropna(subset='Domain',inplace=True)


# In[20]:


df.reset_index(drop=True,inplace=True)


# In[21]:


df.dropna(subset='TLD',inplace=True)


# In[22]:


df.reset_index(drop=True,inplace=True)


# In[23]:


df


# In[24]:


for index in range(len(df)):
    if pd.isna(df['TLDLegitimateProb'][index]):
        df['TLDLegitimateProb'][index] = df['TLDLegitimateProb'].mean()


# In[25]:


for index in range(len(df)):
    if pd.isna(df['URLCharProb'][index]):
        df['URLCharProb'][index] = df['URLCharProb'].mean()


# In[26]:


for index in range(len(df)):
    if pd.isna(df['TLDLength'][index]):
        df['TLDLength'][index] = df['TLDLength'].mode()


# In[27]:


df.isnull().sum()


# In[28]:


for index in range(len(df)):
    if pd.isna(df['LineOfCode'][index]):
        df['LineOfCode'][index] = df['LineOfCode'].mean()


# In[29]:


for index in range(len(df)):
    if pd.isna(df['HasTitle'][index]):
        df['HasTitle'][index] = 1


# In[30]:


for index in range(len(df)):
    if pd.isna(df['URLTitleMatchScore'][index]):
        df['URLTitleMatchScore'][index] = df['URLTitleMatchScore'].median()


# In[31]:


for index in range(len(df)):
    if pd.isna(df['HasFavicon'][index]):
        df['HasFavicon'][index] = df['HasFavicon'].mode()


# In[32]:


df.isnull().sum()


# In[33]:


for index in range(len(df)):
    if pd.isna(df['IsResponsive'][index]):
        df['IsResponsive'][index] = df['IsResponsive'].mode()


# In[34]:


for index in range(len(df)):
    if pd.isna(df['HasDescription'][index]):
        df['HasDescription'][index] = df['HasDescription'].mode()


# In[35]:


for index in range(len(df)):
    if pd.isna(df['HasSocialNet'][index]):
        df['HasSocialNet'][index] = df['HasSocialNet'].mode()


# In[36]:


for index in range(len(df)):
    if pd.isna(df['NoOfPopup'][index]):
        df['NoOfPopup'][index] = df['NoOfPopup'].median()


# In[37]:


df.isnull().sum()


# In[38]:


import matplotlib.pyplot as plt
for col in df.select_dtypes(include=['float64']):
    df[col].plot(kind='hist', color='green', title="Histogram")
    plt.xlabel(col)
    plt.show()


# In[39]:


import matplotlib.pyplot as plt
for col in df.select_dtypes(include=['float64']):
    df[col].plot(kind='box', color='blue', title="Box Plot")
    plt.show()


# In[40]:


import seaborn as sns

for col in df.select_dtypes(include=['float64']):
    sns.scatterplot(x=col, y='label', data=df)
    plt.title('Scatter Plot of Scaled Numerical Columns with target')
    plt.show()


# In[41]:


df


# In[42]:


plt.figure(figsize=(40, 15))  
sns.heatmap(df.select_dtypes(include=['float64','int64']).corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)

plt.title('Correlation Heatmap of Numerical Columns')

plt.show()


# In[43]:


df.dtypes


# In[44]:


df = df.drop('FILENAME',axis=1)
df = df.drop('URL',axis=1)
df = df.drop('Domain',axis=1)
df = df.drop('TLD',axis=1)
df = df.drop('Title',axis=1)

df


# In[45]:


numerical_cols = df.select_dtypes(include=['float64'])

correlations = numerical_cols.corrwith(df['label'])

plt.figure(figsize=(18, 5))
sns.barplot(x=correlations.index, y=correlations.values)

plt.title('Correlation of Each Numerical Column with Target')
plt.xlabel('Columns')
plt.ylabel('Correlation with Target')

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# In[46]:


df.columns


# In[47]:


df = df.drop('URLLength',axis=1)
df = df.drop('TLDLegitimateProb',axis=1)
df = df.drop('TLDLength',axis=1)
df = df.drop('NoOfSubDomain',axis=1)
df = df.drop('NoOfObfuscatedChar',axis=1)
df = df.drop('NoOfDegitsInURL',axis=1)
df = df.drop('NoOfEqualsInURL',axis=1)
df = df.drop('NoOfQMarkInURL',axis=1)
df = df.drop('NoOfAmpersandInURL',axis=1)
df = df.drop('LineOfCode',axis=1)
df = df.drop('LargestLineLength',axis=1)
df = df.drop('NoOfURLRedirect',axis=1)
df = df.drop('NoOfSelfRedirect',axis=1)
df = df.drop('NoOfPopup',axis=1)
df = df.drop('NoOfiFrame',axis=1)
df = df.drop('Bank',axis=1)
df = df.drop('Pay',axis=1)
df = df.drop('Crypto',axis=1)
df = df.drop('NoOfImage',axis=1)
df = df.drop('NoOfCSS',axis=1)
df = df.drop('NoOfJS',axis=1)
df = df.drop('NoOfSelfRef',axis=1)
df = df.drop('NoOfEmptyRef',axis=1)
df = df.drop('NoOfExternalRef',axis=1)
df = df.drop('ObfuscationRatio',axis=1)


# In[48]:


df.columns


# In[49]:


numerical_cols = df.select_dtypes(include=['float64'])

correlations = numerical_cols.corrwith(df['label'])

plt.figure(figsize=(18, 5))
sns.barplot(x=correlations.index, y=correlations.values)

plt.title('Correlation of Each Numerical Column with Target')
plt.xlabel('Columns')
plt.ylabel('Correlation with Target')

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# In[50]:


df.describe()


# In[51]:


df.columns


# In[52]:


df['IsDomainIP'] = df['IsDomainIP'].astype('category')
df['HasObfuscation'] = df['HasObfuscation'].astype('category')
df['IsHTTPS'] = df['IsHTTPS'].astype('category')
df['HasTitle'] = df['HasTitle'].astype('category')
df['HasFavicon'] = df['HasFavicon'].astype('category')
df['IsResponsive'] = df['IsResponsive'].astype('category')
df['HasDescription'] = df['HasDescription'].astype('category')
df['HasExternalFormSubmit'] = df['HasExternalFormSubmit'].astype('category')
df['HasSocialNet'] = df['HasSocialNet'].astype('category')
df['HasSubmitButton'] = df['HasSubmitButton'].astype('category')
df['HasHiddenFields'] = df['HasHiddenFields'].astype('category')
df['HasPasswordField'] = df['HasPasswordField'].astype('category')
df['HasCopyrightInfo'] = df['HasCopyrightInfo'].astype('category')


# In[53]:


df.describe()


# In[54]:


df['label'] = df['label'].astype('category')


# In[55]:


from scipy import stats

col = 'IsDomainIP'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[56]:


df = df.drop('IsDomainIP',axis=1)


# In[57]:


from scipy import stats

col = 'HasObfuscation'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[ ]:





# In[58]:


from scipy import stats

col = 'IsHTTPS'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[ ]:





# In[59]:


from scipy import stats

col = 'HasTitle'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[ ]:





# In[60]:


from scipy import stats

col = 'HasFavicon'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[ ]:





# In[61]:


from scipy import stats

col = 'IsResponsive'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[ ]:





# In[62]:


from scipy import stats

col = 'HasDescription'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[ ]:





# In[63]:


from scipy import stats

col = 'HasExternalFormSubmit'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[ ]:





# In[64]:


from scipy import stats

col = 'HasSocialNet'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[ ]:





# In[65]:


from scipy import stats

col = 'HasSubmitButton'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[ ]:





# In[66]:


from scipy import stats

col = 'HasHiddenFields'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[ ]:





# In[67]:


from scipy import stats

col = 'HasPasswordField'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[ ]:





# In[68]:


from scipy import stats

col = 'HasCopyrightInfo'
epochs = 1000
list = []
for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    contingency_table = pd.crosstab(df_sample[col], df_sample['label'])
    
    chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)
    
    if chi2_stat < 3.841:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

contingency_table = pd.crosstab(df[col], df['label'])
    
chi2_stat, _, dof, _ = stats.chi2_contingency(contingency_table)

print("Validation")
if chi2_stat < 3.841:
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[ ]:





# In[69]:


df['label'] = df['label'].astype('int64')


# In[70]:


df['label']


# In[71]:


phishing = df[df['label']==1]


# In[72]:


non_phishing = df[df['label']==0]


# In[73]:


df.describe()


# In[74]:


from scipy import stats

col = 'DomainLength'
epochs = 1000
list = []

for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    var1 = df_sample[df_sample['label']==0][col].var()*(len(df_sample[df_sample['label']==0]))/(len(df_sample[df_sample['label']==0])-1)
    var2 = df_sample[df_sample['label']==1][col].var()*(len(df_sample[df_sample['label']==1]))/(len(df_sample[df_sample['label']==1])-1)
    
    f_stats = var1/var2
    
    if f_stats < 1:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

print("Validation")
if (df[df['label']==0][col].var())<=(df[df['label']==1][col].var()):
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[75]:


from scipy import stats

col = 'URLSimilarityIndex'
epochs = 1000
list = []

for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    var1 = df_sample[df_sample['label']==0][col].var()*(len(df_sample[df_sample['label']==0]))/(len(df_sample[df_sample['label']==0])-1)
    var2 = df_sample[df_sample['label']==1][col].var()*(len(df_sample[df_sample['label']==1]))/(len(df_sample[df_sample['label']==1])-1)
    
    f_stats = var1/var2
    
    if f_stats < 1:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

print("Validation")
if (df[df['label']==0][col].var())<=(df[df['label']==1][col].var()):
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[76]:


from scipy import stats

col = 'NoOfLettersInURL'
epochs = 1000
list = []

for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    var1 = df_sample[df_sample['label']==0][col].var()*(len(df_sample[df_sample['label']==0]))/(len(df_sample[df_sample['label']==0])-1)
    var2 = df_sample[df_sample['label']==1][col].var()*(len(df_sample[df_sample['label']==1]))/(len(df_sample[df_sample['label']==1])-1)
    
    f_stats = var1/var2
    
    if f_stats < 1:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

print("Validation")
if (df[df['label']==0][col].var())<=(df[df['label']==1][col].var()):
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[77]:


from scipy import stats

col = 'NoOfOtherSpecialCharsInURL'
epochs = 1000
list = []

for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    var1 = df_sample[df_sample['label']==0][col].var()*(len(df_sample[df_sample['label']==0]))/(len(df_sample[df_sample['label']==0])-1)
    var2 = df_sample[df_sample['label']==1][col].var()*(len(df_sample[df_sample['label']==1]))/(len(df_sample[df_sample['label']==1])-1)
    
    f_stats = var1/var2
    
    if f_stats < 1:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

print("Validation")
if (df[df['label']==0][col].var())<=(df[df['label']==1][col].var()):
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[78]:


from scipy import stats

col = 'DomainTitleMatchScore'
epochs = 1000
list = []

for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    var1 = df_sample[df_sample['label']==0][col].var()*(len(df_sample[df_sample['label']==0]))/(len(df_sample[df_sample['label']==0])-1)
    var2 = df_sample[df_sample['label']==1][col].var()*(len(df_sample[df_sample['label']==1]))/(len(df_sample[df_sample['label']==1])-1)
    
    f_stats = var1/var2
    
    if f_stats < 1:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

print("Validation")
if (df[df['label']==0][col].var())<=(df[df['label']==1][col].var()):
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[79]:


from scipy import stats

col = 'URLTitleMatchScore'
epochs = 1000
list = []

for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    var1 = df_sample[df_sample['label']==0][col].var()*(len(df_sample[df_sample['label']==0]))/(len(df_sample[df_sample['label']==0])-1)
    var2 = df_sample[df_sample['label']==1][col].var()*(len(df_sample[df_sample['label']==1]))/(len(df_sample[df_sample['label']==1])-1)
    
    f_stats = var1/var2
    
    if f_stats < 1:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

print("Validation")
if (df[df['label']==0][col].var())<=(df[df['label']==1][col].var()):
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[80]:


from scipy import stats

col = 'CharContinuationRate'
epochs = 1000
list = []

for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    non_phishing = df[df['label']==0]
    phishing = df[df['label']==1]
    
    if len(non_phishing) < 30 or len(phishing) < 30:
        continue
    
    mean1 = non_phishing[col].mean()
    mean2 = phishing[col].mean()
    
    var1 = non_phishing[col].var()
    var2 = phishing[col].var()
    
    z_stats = (mean1 - mean2)/(((var1/len(non_phishing[col])) + (var2/len(phishing[col])))**(1/2))
    
    if  z_stats < 1.645:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

print("Validation")
if (df[df['label']==0][col].mean())<=(df[df['label']==1][col].mean()):
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[81]:


from scipy import stats

col = 'URLCharProb'
epochs = 1000
list = []

for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    non_phishing = df[df['label']==0]
    phishing = df[df['label']==1]
    
    if len(non_phishing) < 30 or len(phishing) < 30:
        continue
    
    mean1 = non_phishing[col].mean()
    mean2 = phishing[col].mean()
    
    var1 = non_phishing[col].var()
    var2 = phishing[col].var()
    
    z_stats = (mean1 - mean2)/(((var1/len(non_phishing[col])) + (var2/len(phishing[col])))**(1/2))
    
    if  z_stats < 1.645:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

print("Validation")
if (df[df['label']==0][col].mean())<=(df[df['label']==1][col].mean()):
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[82]:


from scipy import stats

col = 'LetterRatioInURL'
epochs = 1000
list = []

for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    non_phishing = df[df['label']==0]
    phishing = df[df['label']==1]
    
    if len(non_phishing) < 30 or len(phishing) < 30:
        continue
    
    mean1 = non_phishing[col].mean()
    mean2 = phishing[col].mean()
    
    var1 = non_phishing[col].var()
    var2 = phishing[col].var()
    
    z_stats = (mean1 - mean2)/(((var1/len(non_phishing[col])) + (var2/len(phishing[col])))**(1/2))
    
    if  z_stats < 1.645:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

print("Validation")
if (df[df['label']==0][col].mean())<=(df[df['label']==1][col].mean()):
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[83]:


from scipy import stats

col = 'DegitRatioInURL'
epochs = 1000
list = []

for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    non_phishing = df[df['label']==0]
    phishing = df[df['label']==1]
    
    if len(non_phishing) < 30 or len(phishing) < 30:
        continue
    
    mean1 = non_phishing[col].mean()
    mean2 = phishing[col].mean()
    
    var1 = non_phishing[col].var()
    var2 = phishing[col].var()
    
    z_stats = (mean1 - mean2)/(((var1/len(non_phishing[col])) + (var2/len(phishing[col])))**(1/2))
    
    if  z_stats < 1.645:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

print("Validation")
if (df[df['label']==0][col].mean())<=(df[df['label']==1][col].mean()):
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[84]:


from scipy import stats

col = 'SpacialCharRatioInURL'
epochs = 1000
list = []

for i in range(epochs):
    df_sample = df.sample(n=20000,replace=False)
    
    non_phishing = df[df['label']==0]
    phishing = df[df['label']==1]
    
    if len(non_phishing) < 30 or len(phishing) < 30:
        continue
    
    mean1 = non_phishing[col].mean()
    mean2 = phishing[col].mean()
    
    var1 = non_phishing[col].var()
    var2 = phishing[col].var()
    
    z_stats = (mean1 - mean2)/(((var1/len(non_phishing[col])) + (var2/len(phishing[col])))**(1/2))
    
    if  z_stats < 1.645:
        list.append('true')
    else:
        list.append('false')

hyp_result = max(list)
print("Random Sample Majority : "+ hyp_result)

print("Validation")
if (df[df['label']==0][col].mean())<=(df[df['label']==1][col].mean()):
    print("Whole sample"+ 'true')
    if hyp_result == 'true':
        print("Correct")
    else:
        print("Incorrect")
else:
    print("Whole Sample : "+ 'false')
    if hyp_result == 'false':
        print("Correct")
    else:
        print("Incorrect")


# In[85]:


df['HasObfuscation'] = df['HasObfuscation'].astype('int')
df['IsHTTPS'] = df['IsHTTPS'].astype('int')
df['HasTitle'] = df['HasTitle'].astype('int')
df['HasFavicon'] = df['HasFavicon'].astype('int')
df['IsResponsive'] = df['IsResponsive'].astype('int')
df['HasDescription'] = df['HasDescription'].astype('int')
df['HasExternalFormSubmit'] = df['HasExternalFormSubmit'].astype('int')
df['HasSocialNet'] = df['HasSocialNet'].astype('int')
df['HasSubmitButton'] = df['HasSubmitButton'].astype('int')
df['HasHiddenFields'] = df['HasHiddenFields'].astype('int')
df['HasPasswordField'] = df['HasPasswordField'].astype('int')
df['HasCopyrightInfo'] = df['HasCopyrightInfo'].astype('int')


# In[86]:


df.columns


# In[87]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import time  # For measuring execution time


# Step 1: Separate features and target
X = df.drop(columns=['label'])
y = df['label']

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate models
for name, model in models.items():
    # Measure training time
    train_start = time.time()
    model.fit(X_train, y_train)
    train_end = time.time()
    
    # Measure evaluation time
    eval_start = time.time()
    y_pred = model.predict(X_test)
    eval_end = time.time()
    
    # Print results
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Training Time: {train_end - train_start:.6f} seconds")
    print(f"Evaluation Time: {eval_end - eval_start:.6f} seconds")
    print("-" * 30)


# In[88]:


df.dtypes


# In[89]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
import time

# Separate features and target
X = df.drop(columns=['label'])
y = df['label']

# Augment data with random noise
def add_random_noise(data, noise_level=0.1):
    noise = noise_level * np.random.normal(size=data.shape)
    return data + noise

X_augmented = add_random_noise(X)

# Combine augmented data with original
X_combined = pd.concat([X, pd.DataFrame(X_augmented, columns=X.columns)], ignore_index=True)
y_combined = pd.concat([y, y], ignore_index=True)

# Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_combined, y_combined)

# Apply SVD
n_components = min(X_balanced.shape[1] - 1, 25)  # Set maximum components (e.g., 50 or less than total features)
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(X_balanced)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_balanced, test_size=0.2, random_state=42)

# Measure training and evaluation times
train_start = time.time()
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
train_end = time.time()

eval_start = time.time()
y_pred = model.predict(X_test)
eval_end = time.time()

# Print Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print timing information
print("\nModel Training and Evaluation Times:")
print(f"Model Training Time: {train_end - train_start:.6f} seconds")
print(f"Model Evaluation Time: {eval_end - eval_start:.6f} seconds")


# In[ ]:



