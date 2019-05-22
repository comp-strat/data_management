
# coding: utf-8

# In[1]:


import pandas as pd 
import pickle as pk
import numpy as np


# In[22]:



files = np.array([])
for i in range(17):
    if i==13:
        continue;
    if i>= 10:
        files = np.append(files, pd.read_csv(str(i)+"f.csv", encoding = "latin1", low_memory = False))
    else:
        files = np.append(files, pd.read_csv("0"+str(i)+"f.csv", encoding = "latin1",low_memory = False))


# In[25]:


len(files)


# In[27]:


files[0]


# In[2]:


#Reads each csv file and coverts it into a 2-D data structure (Dataframe).
f00 = pd.read_csv("00f.csv", encoding = "latin1", low_memory = False)
f01 = pd.read_csv("01f.csv", encoding = 'latin1', low_memory = False)
f02 = pd.read_csv("02f.csv", encoding = 'latin1', low_memory = False)
f03 = pd.read_csv("03f.csv", encoding = 'latin1', low_memory = False)
f04 = pd.read_csv("04f.csv", encoding = 'latin1', low_memory = False)
f05 = pd.read_csv("05f.csv", encoding = 'latin1', low_memory = False)
f06 = pd.read_csv("06f.csv", encoding = 'latin1', low_memory = False)
f07 = pd.read_csv("07f.csv", encoding = 'latin1', low_memory = False)
f08 = pd.read_csv("08f.csv", encoding = 'latin1', low_memory = False)
f09 = pd.read_csv("09f.csv", encoding = 'latin1', low_memory = False)
f10 = pd.read_csv("10f.csv", encoding = 'latin1', low_memory = False)
f11 = pd.read_csv("11f.csv", encoding = 'latin1', low_memory = False)
f12 = pd.read_csv("12f.csv", encoding = 'latin1', low_memory = False)
f14 = pd.read_csv("14f.csv", encoding = 'latin1', low_memory = False)
f15 = pd.read_csv("15f.csv", encoding = 'latin1', low_memory = False)
f16 = pd.read_csv("16f.csv", encoding = 'latin1', low_memory = False)
f98 = pd.read_csv("98f.csv", encoding = 'latin1', low_memory = False)


# In[37]:


#List of column names of some random files 
print(len(list(f05)))
print(len(list(f06)))


# In[3]:


# For the how argument we are using 'outer'. Outer keeps all the unmatched rows for both sides of the merge file
mergeddf = pd.merge(f98,f00, how= 'outer', on = 'NCESSCH')
mergeddf = pd.merge(mergeddf,f01, how= 'outer', on = 'NCESSCH')
mergeddf = pd.merge(mergeddf,f02, how = 'outer', on = 'NCESSCH')
mergeddf = pd.merge(mergeddf,f03, how = 'outer', on= "NCESSCH")
mergeddf = pd.merge(mergeddf,f04, how = 'outer', on = "NCESSCH")
mergeddf = pd.merge(mergeddf,f05, how = 'outer', on = "NCESSCH")
mergeddf = pd.merge(mergeddf,f06, how = 'outer', on = "NCESSCH")
mergeddf = pd.merge(mergeddf,f07, how = 'outer', on = "NCESSCH")
mergeddf = pd.merge(mergeddf,f08, how = 'outer', on = "NCESSCH")
mergeddf = pd.merge(mergeddf,f09, how = 'outer', on = "NCESSCH")
mergeddf = pd.merge(mergeddf,f10, how = 'outer', on = "NCESSCH")
mergeddf = pd.merge(mergeddf,f11, how = 'outer', on = "NCESSCH")
mergeddf = pd.merge(mergeddf,f12, how = 'outer', on = "NCESSCH")
mergeddf = pd.merge(mergeddf,f14, how = 'outer', on = "NCESSCH")
mergeddf = pd.merge(mergeddf,f15, how = 'outer', on = "NCESSCH")
mergeddf = pd.merge(mergeddf,f16, how = 'outer', on = "NCESSCH")
mergeddf 



# In[6]:


print(len(f98))
print(len(f98['NCESSCH'].unique()))


# In[7]:


mergeddf = pd.merge(f98,f00, how= 'outer', on = 'NCESSCH')
mergeddf = pd.merge(mergeddf,f01, how= 'outer', on = 'NCESSCH')


# In[8]:


print(len(f98))
print(len(f00))
print(len(f01))
print(len(mergeddf))

