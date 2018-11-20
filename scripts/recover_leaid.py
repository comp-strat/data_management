# coding: utf-8

# Recover LEAID by geographical info
 
# Authors: Ji Shi, Jiahua Zou, Jaren Haber
# Institution: UC Berkeley
# Contact: jhaber@berkeley.edu
 
# Created: Nov. 12, 2018
# Last modified: Nov. 19, 2018

# Description: Geographically identifies each public school's LEAID (Local Education Agency Identification Code) by locating that school's latitude and longitude coordinates within school district shapefiles. Given that NCES school data lists for many schools--especially charter schools--an LEAID with legal but not geographic significance, this geographic matching allows analysis of each school within the community (school district) context in which it is physically situated. 


# Initialize

import pandas as pd
import numpy as np
import os
import gc # Makes loading pickle files faster 
import sys # Script tricks (like exiting)

from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
tqdm.pandas(desc="Matching coords->LEAIDs") # To show progress, create & register new `tqdm` instance with `pandas`

try:
    from shapely.geometry import Point, Polygon
except ImportError:
    get_ipython().system("pip install shapely # if module doesn't exist, then install")
    from shapely.geometry import Point, Polygon

try:
    import geopandas as gpd
except ImportError:
    get_ipython().system("pip install geopandas # if module doesn't exist, then install")
    import geopandas as gpd


# Define helper functions

def quickpickle_load(picklepath):
    '''Very time-efficient way to load pickle-formatted objects into Python.
    Uses C-based pickle (cPickle) and gc workarounds to facilitate speed. 
    Input: Filepath to pickled (*.pkl) object.
    Output: Python object (probably a list of sentences or something similar).'''

    with open(picklepath, 'rb') as loadfile:
        
        gc.disable() # disable garbage collector
        outputvar = cPickle.load(loadfile) # Load from picklepath into outputvar
        gc.enable() # enable garbage collector again
    
    return outputvar


def quickpickle_dump(dumpvar, picklepath):
    '''Very time-efficient way to dump pickle-formatted objects from Python.
    Uses C-based pickle (cPickle) and gc workarounds to facilitate speed. 
    Input: Python object (probably a list of sentences or something similar).
    Output: Filepath to pickled (*.pkl) object.'''

    with open(picklepath, 'wb') as destfile:
        
        gc.disable() # disable garbage collector
        cPickle.dump(dumpvar, destfile) # Dump dumpvar to picklepath
        gc.enable() # enable garbage collector again
        

# Load data
    
print("Loading data...")
chartersdf = quickpickle_load("../../nowdata/charters_2015.pkl") # Charter school data (mainly NCES CCD PSUS, 2015-16)
#pubsdf = quickpickle_load("../../nowdata/pubschools_2015.pkl") # Charter school data (mainly NCES CCD PSUS, 2015-16)
sddata = quickpickle_load("../data/US_sd_combined_2016.pkl") # School district shapefiles (2016)
acs = pd.read_csv("../data/ACS_2016_sd-merged_FULL.csv", header = [0, 1], encoding="latin1", low_memory=False) # School district social data (ACS, 2012-16)

# Save school district shapes & identifiers for use later:
sdshapes = sddata["geometry"]
sdid = sddata["FIPS"]        
        
        
# Define core function
        
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
        
        # Note: LAT & LON coordinates appear reversed when invoked with Point for gpd()
        if sdshapes[i].contains(Point((cord[1], cord[0]))): 
            #print('Find a school district!')
            if sdid.loc[i] != original:
                #print(sdid.loc[i], "vs.", original, "- changed!")
                return sdid.loc[i]
            #print(sdid.loc[i], "vs.", original)
            return sdid.loc[i]
    return original


# Execute core function

#pubsdf['GEO_LEAID'] = pubsdf[["LAT1516", 'LON1516','LEAID']].progress_apply(lambda x: mapping_leaid((x["LAT1516"], x['LON1516']), x['LEAID']), axis=1)
chartersdf['GEO_LEAID'] = chartersdf[["LAT1516", 'LON1516','LEAID']].progress_apply(lambda x: mapping_leaid((x["LAT1516"], x['LON1516']), x['LEAID']), axis=1)


# Merge ACS data again using new LEAID

print("Merging ACS data using new 'GEO_LEAID'...")

# Get list of ACS vars to drop from school dfs (to avoid risk of duplication):
acsvars_ch, acsvars_pub = [], []
for var in list(acs):
    if var[1] in list(pubsdf):
        acsvars_ch.append(var[1])
    if var[1] in list(chartersdf):
        acsvars_pub.append(var[1])

acs["GEO_LEAID"] = acs[("FIPS", "Geo_FIPS")] # Simplifies merging process
chartersdf = chartersdf.drop(acsvars_ch, axis=1) # Drop ACS vars
chartersdf = pd.merge(chartersdf, acs, how="left", on="GEO_LEAID")
#pubsdf = pubsdf.drop(acsvars_pub, axis=1) # Drop ACS vars
#pubsdf = pd.merge(pubsdf, acs, how="left", on="GEO_LEAID")


# Save modified data to disk

print("Saving data to disk...")
#quickpickle_dump(pubsdf, "../nowdata/backups/charters_full_2015_250_v2a_unlappedtext_counts3_geoleaid.pkl")
quickpickle_dump(chartersdf, "../nowdata/backups/pubschools_full_2015_v2a_geoleaid.pkl")
print("...done.")

sys.exit() # Exit script (to be safe)