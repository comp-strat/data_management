
# coding: utf-8

# # Recover LEAID by geographical info
# 
# Authors: Ji Shi, Jiahua Zou, Jaren Haber <br>
# Institution: UC Berkeley <br>
# Contact: jhaber@berkeley.edu
# 
# Created: Nov. 12, 2018 <br>
# Last modified: Nov. 19, 2018
# 
# Description: Geographically identifies each public school's LEAID (Local Education Agency Identification Code) by locating that school's latitude and longitude coordinates within school district shapefiles. Given that NCES school data lists for many schools--especially charter schools--an LEAID with legal but not geographic significance, this geographic matching allows analysis of each school within the community (school district) context in which it is physically situated. 

# ## Initialize

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


try:
    from shapely.geometry import Point, Polygon
except ImportError:
    get_ipython().system("pip install shapely # if module doesn't exist, then install")
    from shapely.geometry import Point, Polygon


# In[3]:


try:
    import geopandas as gpd
except ImportError:
    get_ipython().system("pip install geopandas # if module doesn't exist, then install")
    import geopandas as gpd


# In[4]:


import gc # Makes loading pickle files faster 
gc.disable() # Disable garbage collector
char_sch = pd.read_pickle("../nowdata/charters_2015.pkl")
gc.enable() # Re-enable garbage collector


# In[5]:


# Inspect the file:
print(char_sch.shape)
print(list(char_sch))
char_sch


# In[39]:


# These latitude & longitude coordinates do correspond to the matching address:
char_sch[["ADDRESS1516", "LAT1516", "LON1516"]][:20]


# In[7]:


cf = pd.read_pickle("districtchangedlol.pkl")


# In[8]:


#check if geometry type is Polygon
cf.loc[0, "geometry"]


# In[33]:


# Save school district shapes, identifiers for use later:
sdshapes = cf["geometry"]
sdid = cf["FIPS"]
sdshapes[0]


# In[34]:


sdshapes[:5]


# In[35]:


sdid[:5]


# In[27]:


# Load ACS file:
acs = pd.read_csv("../data_management/data/ACS_2016_sd-merged_FULL.csv", header = [0, 1], encoding="latin1", low_memory=False)

# Inspect the file:
print(acs.shape)
print(list(acs))
acs["FIPS"].astype(int)


# In[28]:


print(char_sch['LEAID'].dropna().apply(lambda x: int(x)))
print(acs.FIPS.astype(int))


# In[29]:


#We want to recover the LEAID that do not appear in ACS_2016_sd-merged_FULL.csv
#Turns out all LEAID in charters15 do not appear in ACS_2016_sd-merged_FULL.csv
#So we have to recover all of them

char_sch["in_acs"] = char_sch['LEAID'].dropna().apply(lambda x: int(x) in acs.FIPS.astype(int))
print(sum(char_sch["in_acs"]))


# In[51]:


def mapping_leaid(cord, original):
    '''This method takes in the coordinates of a certain school and search for all 
       school districts to see whether there is a school district contain the coordinates 
       by using Polygon.contains. If so, return the coresponding LEAID else return the 
       original LEAID
       
       Args:
           cord: tuple of lat and long
           original: the original LEAID
       Returns:
           LEAID
    '''
    
    global sdshapes, sdid # School district polygons, identifiers (equivalent to LEAID)
    
    if not cord[0] or not cord[1]:
        return original
    #print("Processing", cord)

    for i in range(len(sdshapes)):
        
#         if cf.loc[i, "FIPS"] == original:
#             print(i)
#             print(cf.loc[i, "FIPS"])
#             print(sdshapes[i].contains(Point((cord[1], cord[0]))))

        #LAT and LONG indeed reversed!
        #fixed below
        if sdshapes[i].contains(Point((cord[1], cord[0]))):
            #print('Find a school district!')
            if sdid.loc[i] != original:
                print(sdid.loc[i], "vs.", original, "- changed!")
                return sdid.loc[i]
            print(sdid.loc[i], "vs.", original)
            return sdid.loc[i]
    return original


# In[59]:


#Choose some schools to test differences between original LEAID and newly found LEAID
char_sch.loc[0:100, ['LAT1516', 'LON1516','LEAID']].apply(lambda x: mapping_leaid((x["LAT1516"],x['LON1516']), x['LEAID']), axis=1)


# In[52]:


# check randomly picked 401460.0 vs. 400016.0
# The original schools labeled 400016.0
char_sch[char_sch['LEAID'] == 400016.0][['URL','SCH_NAME', 'LAT1516', 'LON1516']]


# In[52]:


#The school district found by our algorithm
cf[cf['FIPS'] == 401460.0]


# In[54]:


#The school district of the original label 400016.0
cf[cf['FIPS'] == 400016.0]


# **_I checked the google map and found that the five schools above are indeed physically in the school district found by our algorithm. But I don't know why our original LEAID (in this case 401460) does not correspond to any school district._**

# In[57]:


print(char_sch.shape)
list(char_sch)


# In[62]:


char_sch[char_sch["LEAID"] == 400016.0][["LEAID", "URL", "LAT1516", "LON1516", "LOCALE15", "ADDRESS16"]]


# In[ ]:


#Please run again.
char_sch['GEO_LEAID'] = char_sch[["LAT1516", 'LON1516','LEAID']].apply(lambda x: mapping_leaid((x["LAT1516"],x['LON1516']), x['LEAID']), axis=1)


# In[58]:


# Check out the resulting DF:
print(char_sch[["LAT1516","LON1516", "LEAID", "GEO_LEAID"]].isna().apply(sum))
char_sch[["LAT1516","LON1516", "LEAID", "GEO_LEAID"]]


# In[42]:


char_sch.to_pickle("../nowdata/backups/charters_full_2015_250_v2a_unlappedtext_counts3_geoleaid.pkl")


# ## Prep for multiprocessing (probably not necessary)

# In[ ]:


# Import packages for multiprocessing
get_ipython().system('pip install tqdm # for monitoring progress of multiprocessing')
numcpus = len(os.sched_getaffinity(0)) # Detect and assign number of available CPUs
from multiprocessing import Pool # key function for multiprocessing, to increase processing speed
pool = Pool(processes=numcpus) # Pre-load number of CPUs into pool function


# In[24]:


tuplist = [tuple(x) for x in char_sch[["NCESSCH", "LAT1516", "LON1516", "LEAID"]].values]
tuplist


# In[ ]:


# Use multiprocessing.Pool(numcpus) to run your function:
print("Matching schools with LEAID based on LAT/LON coordinates:
if __name__ == '__main__':
    with Pool(numcpus) as p:
        p.map(mapping_leaid(), tqdm(tuplist, desc="Matching LAT/LON to LEAID")) 

