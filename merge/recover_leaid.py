# coding: utf-8

# Recover LEAID by geographical info
 
# Authors: Jaren Haber, Ji Shi, Jiahua Zou
# Institution: UC Berkeley
# Contact: jhaber@berkeley.edu
 
# Created: Nov. 12, 2018
# Last modified: Nov. 19, 2018

# Description: Geographically identifies each public school's LEAID (Local Education Agency Identification Code) by locating that school's latitude and longitude coordinates within school district shapefiles. Given that NCES school data lists for many schools--especially charter schools--an LEAID with legal but not geographic significance, the geographic matching here allows analysis of each school within the community (school district) context in which it is physically and socially situated. 


# Initialize

import pandas as pd
import gc # Makes loading pickle files faster 
import sys # Script tricks (like exiting)
import _pickle as cPickle # Optimized version of pickle

from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
tqdm.pandas(desc="Matching coords-->LEAIDs") # To show progress, create & register new `tqdm` instance with `pandas`

try:
    from shapely.geometry import Point, Polygon
except ImportError:
    get_ipython().system("pip install shapely") # if module doesn't exist, then install
    from shapely.geometry import Point, Polygon

try:
    import geopandas as gpd
except ImportError:
    get_ipython().system("pip install geopandas") # if module doesn't exist, then install
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
#chartersdf = quickpickle_load("../../nowdata/charters_2015.pkl") # Charter school data (mainly NCES CCD PSUS, 2015-16)
pubsdf = quickpickle_load("../../nowdata/pubschools_2015.pkl") # Charter school data (mainly NCES CCD PSUS, 2015-16)
sddata = quickpickle_load("../data/US_sd_combined_2016.pkl") # School district shapefiles (2016)
acs = pd.read_csv("../data/ACS_2016_sd-merged_FULL.csv", header = [0, 1], encoding="latin1", low_memory=False) # School district social data (ACS, 2012-16)

# Save school district polygons/shapes & identifiers for use later:
sdshapes = sddata["geometry"]
sdid = sddata["FIPS"]        
        
        
# Define core function
        
def mapping_leaid(cord, original):
    '''This method takes in the coordinates of a certain school and search for all 
       school districts to see whether there is a school district contain the coordinates 
       by using Polygon.contains. If so, return the corresponding GEO_LEAID else return the 
       original LEAID
       
       Args:
           cord: tuple of lat and long
           original: the original LEAID
       Returns:
           GEO_LEAID
    '''
    
    global sdshapes, sdid # School district polygons, identifiers (equivalent to LEAID)
    
    # If no coordinates, then can't match -> return original LEAID
    if not cord[0] or not cord[1]:
        return original

    for i in range(len(sdshapes)): # Look at all school districts for matches
        
        # Note: LAT & LON coordinates appear reversed when invoked with Point for gpd(), hence the order below
        # Locate school's coordinates geographically within school district:
        if sdshapes[i].contains(Point((cord[1], cord[0]))): 
            return sdid.loc[i]
        
    # If no match, return original LEAID
    return original


# Execute core function

#chartersdf['GEO_LEAID'] = chartersdf[["LAT1516", 'LON1516','LEAID']].progress_apply(lambda x: mapping_leaid((x["LAT1516"], x['LON1516']), x['LEAID']), axis=1)
pubsdf['GEO_LEAID'] = pubsdf[["LAT1516", 'LON1516','LEAID']].progress_apply(lambda x: mapping_leaid((x["LAT1516"], x['LON1516']), x['LEAID']), axis=1)


# Merge ACS data again using new LEAID

print("Merging ACS data using new 'GEO_LEAID'...")

# Get list of ACS vars to drop from school dfs (to avoid risk of duplication):
acsvars_ch, acsvars_pub = [], []
for var in list(acs):
    #if var[1] in list(chartersdf) and var[1] not in acsvars_ch:
    #    acsvars_ch.append(var[1])
    if var[1] in list(pubsdf) and var[1] not in acsvars_pub:
        acsvars_pub.append(var[1])
    if var in list(pubsdf) and var not in acsvars_pub:
        acsvars_pub.append(var)


acs["GEO_LEAID"] = acs[("FIPS", "Geo_FIPS")] # Simplifies merging process
#chartersdf = chartersdf.drop(acsvars_ch, axis=1) # Drop ACS vars
#chartersdf = pd.merge(chartersdf, acs, how="left", on="GEO_LEAID")
pubsdf = pubsdf.drop(acsvars_pub, axis=1) # Drop ACS vars
pubsdf = pd.merge(pubsdf, acs, how="left", on="GEO_LEAID")

# Rename ACS columns back to original coding (so STATA can work with them)
# Create tuples for naming scheme: (acs_code, acs_words)
acs_tups = ('FIPS', 'Geo_FIPS'), ('Name of Area', 'Geo_NAME'), ('Qualifying Name', 'Geo_QName'), ('State/U.S.-Abbreviation (USPS)', 'Geo_STUSAB'), ('Summary Level', 'Geo_SUMLEV'), ('Geographic Component', 'Geo_GEOCOMP'), ('File Identification', 'Geo_FILEID'), ('Logical Record Number', 'Geo_LOGRECNO'), ('US', 'Geo_US'), ('Region', 'Geo_REGION'), ('Division', 'Geo_DIVISION'), ('State (Census Code)', 'Geo_STATECE'), ('State (FIPS)', 'Geo_STATE'), ('County', 'Geo_COUNTY'), ('County Subdivision (FIPS)', 'Geo_COUSUB'), ('Place (FIPS Code)', 'Geo_PLACE'), ('Place (State FIPS + Place FIPS)', 'Geo_PLACESE'), ('Census Tract', 'Geo_TRACT'), ('Block Group', 'Geo_BLKGRP'), ('Consolidated City', 'Geo_CONCIT'), ('American Indian Area/Alaska Native Area/Hawaiian Home Land (Census)', 'Geo_AIANHH'), ('American Indian Area/Alaska Native Area/Hawaiian Home Land (FIPS)', 'Geo_AIANHHFP'), ('American Indian Trust Land/Hawaiian Home Land Indicator', 'Geo_AIHHTLI'), ('American Indian Tribal Subdivision (Census)', 'Geo_AITSCE'), ('American Indian Tribal Subdivision (FIPS)', 'Geo_AITS'), ('Alaska Native Regional Corporation (FIPS)', 'Geo_ANRC'), ('Metropolitan and Micropolitan Statistical Area', 'Geo_CBSA'), ('Combined Statistical Area', 'Geo_CSA'), ('Metropolitan Division', 'Geo_METDIV'), ('Metropolitan Area Central City', 'Geo_MACC'), ('Metropolitan/Micropolitan Indicator Flag', 'Geo_MEMI'), ('New England City and Town Combined Statistical Area', 'Geo_NECTA'), ('New England City and Town Area', 'Geo_CNECTA'), ('New England City and Town Area Division', 'Geo_NECTADIV'), ('Urban Area', 'Geo_UA'), ('Urban Area Central Place', 'Geo_UACP'), ('Current Congressional District ***', 'Geo_CDCURR'), ('State Legislative District Upper', 'Geo_SLDU'), ('State Legislative District Lower', 'Geo_SLDL'), ('Voting District', 'Geo_VTD'), ('ZIP Code Tabulation Area (3-digit)', 'Geo_ZCTA3'), ('ZIP Code Tabulation Area (5-digit)', 'Geo_ZCTA5'), ('Subbarrio (FIPS)', 'Geo_SUBMCD'), ('School District (Elementary)', 'Geo_SDELM'), ('School District (Secondary)', 'Geo_SDSEC'), ('School District (Unified)', 'Geo_SDUNI'), ('Urban/Rural', 'Geo_UR'), ('Principal City Indicator', 'Geo_PCI'), ('Traffic Analysis Zone', 'Geo_TAZ'), ('Urban Growth Area', 'Geo_UGA'), ('Public Use Microdata Area - 5% File', 'Geo_PUMA5'), ('Public Use Microdata Area - 1% File', 'Geo_PUMA1'), ('Geographic Identifier', 'Geo_GEOID'), ('Tribal Tract', 'Geo_BTTR'), ('Tribal Block Group', 'Geo_BTBG'), ('Area (Land)', 'Geo_AREALAND'), ('Area (Water)', 'Geo_AREAWATR'), ('Total Population', 'SE_T002_001'), ('Population Density (Per Sq. Mile)', 'SE_T002_002'), ('Area (Land)', 'SE_T002_003'), ('Total Population:', 'SE_T013_001'), ('Total Population: White Alone', 'SE_T013_002'), ('Total Population: Black or African American Alone', 'SE_T013_003'), ('Total Population: American Indian and Alaska Native Alone', 'SE_T013_004'), ('Total Population: Asian Alone', 'SE_T013_005'), ('Total Population: Native Hawaiian and Other Pacific Islander Alone', 'SE_T013_006'), ('Total Population: Some Other Race Alone', 'SE_T013_007'), ('Total Population: Two or More Races', 'SE_T013_008'), ('% Total Population: White Alone', 'PCT_SE_T013_002'), ('% Total Population: Black or African American Alone', 'PCT_SE_T013_003'), ('% Total Population: American Indian and Alaska Native Alone', 'PCT_SE_T013_004'), ('% Total Population: Asian Alone', 'PCT_SE_T013_005'), ('% Total Population: Native Hawaiian and Other Pacific Islander Alone', 'PCT_SE_T013_006'), ('% Total Population: Some Other Race Alone', 'PCT_SE_T013_007'), ('% Total Population: Two or More Races', 'PCT_SE_T013_008'), ('Population 25 Years and Over:', 'SE_T025_001'), ('Population 25 Years and Over: Less than High School', 'SE_T025_002'), ('Population 25 Years and Over: High School Graduate (Includes Equivalency)', 'SE_T025_003'), ('Population 25 Years and Over: Some College', 'SE_T025_004'), ("Population 25 Years and Over: Bachelor's Degree", 'SE_T025_005'), ("Population 25 Years and Over: Master's Degree", 'SE_T025_006'), ('Population 25 Years and Over: Professional School Degree', 'SE_T025_007'), ('Population 25 Years and Over: Doctorate Degree', 'SE_T025_008'), ('% Population 25 Years and Over: Less than High School', 'PCT_SE_T025_002'), ('% Population 25 Years and Over: High School Graduate (Includes Equivalency)', 'PCT_SE_T025_003'), ('% Population 25 Years and Over: Some College', 'PCT_SE_T025_004'), ("% Population 25 Years and Over: Bachelor's Degree", 'PCT_SE_T025_005'), ("% Population 25 Years and Over: Master's Degree", 'PCT_SE_T025_006'), ('% Population 25 Years and Over: Professional School Degree', 'PCT_SE_T025_007'), ('% Population 25 Years and Over: Doctorate Degree', 'PCT_SE_T025_008'), ('Population 3 Years and Over:', 'SE_T028_001'), ('Population 3 Years and Over: Enrolled in School', 'SE_T028_002'), ('Population 3 Years and Over: Not Enrolled in School', 'SE_T028_003'), ('% Population 3 Years and Over: Enrolled in School', 'PCT_SE_T028_002'), ('% Population 3 Years and Over: Not Enrolled in School', 'PCT_SE_T028_003'), ('Civilian Population 16 to 19 Years:', 'SE_T030_001'), ('Civilian Population 16 to 19 Years: Not High School Graduate, Not Enrolled (Dropped Out)', 'SE_T030_002'), ('Civilian Population 16 to 19 Years: High School Graduate, or Enrolled (in School)', 'SE_T030_003'), ('% Civilian Population 16 to 19 Years: Not High School Graduate, Not Enrolled (Dropped Out)', 'PCT_SE_T030_002'), ('% Civilian Population 16 to 19 Years: High School Graduate, or Enrolled (in School)', 'PCT_SE_T030_003'), ('Civilian Population in Labor Force 16 Years and Over:', 'SE_T037_001'), ('Civilian Population in Labor Force 16 Years and Over: Employed', 'SE_T037_002'), ('Civilian Population in Labor Force 16 Years and Over: Unemployed', 'SE_T037_003'), ('% Civilian Population in Labor Force 16 Years and Over: Employed', 'PCT_SE_T037_002'), ('% Civilian Population in Labor Force 16 Years and Over: Unemployed', 'PCT_SE_T037_003'), ('Median Household Income (In 2016 Inflation Adjusted Dollars)', 'SE_T057_001'), ('Gini Index', 'SE_T157_001'), ('Families:', 'SE_T113_001'), ('Families: Income in Below Poverty Level', 'SE_T113_002'), ('Families: Income in Below Poverty Level: Married Couple Family: with Related Child Living  Bellow Poverty Level', 'SE_T113_003'), ('Families: Income in Below Poverty Level: Married Couple Family: No Related Children Under 18 Years', 'SE_T113_004'), ('Families: Income in Below Poverty Level: Male Householder, No Wife Present', 'SE_T113_005'), ('Families: Income in Below Poverty Level: Male Householder, No Wife Present: with Related Children Under 18 Years', 'SE_T113_006'), ('Families: Income in Below Poverty Level: Male Householder, No Wife Present: No Related Children Under 18 Years', 'SE_T113_007'), ('Families: Income in Below Poverty Level: Female Householder, No Husband Present', 'SE_T113_008'), ('Families: Income in Below Poverty Level: Female Householder, No Husband Present: with Related Children Under 18 Years', 'SE_T113_009'), ('Families: Income in Below Poverty Level: Female Householder, No Husband Present: No Related Children Under 18 Years', 'SE_T113_010'), ('Families: Income in at or Above Poverty Level', 'SE_T113_011'), ('% Families: Income in Below Poverty Level', 'PCT_SE_T113_002'), ('% Families: Income in Below Poverty Level: Married Couple Family: with Related Child Living  Bellow Poverty Level', 'PCT_SE_T113_003'), ('% Families: Income in Below Poverty Level: Married Couple Family: No Related Children Under 18 Years', 'PCT_SE_T113_004'), ('% Families: Income in Below Poverty Level: Male Householder, No Wife Present', 'PCT_SE_T113_005'), ('% Families: Income in Below Poverty Level: Male Householder, No Wife Present: with Related Children Under 18 Years', 'PCT_SE_T113_006'), ('% Families: Income in Below Poverty Level: Male Householder, No Wife Present: No Related Children Under 18 Years', 'PCT_SE_T113_007'), ('% Families: Income in Below Poverty Level: Female Householder, No Husband Present', 'PCT_SE_T113_008'), ('% Families: Income in Below Poverty Level: Female Householder, No Husband Present: with Related Children Under 18 Years', 'PCT_SE_T113_009'), ('% Families: Income in Below Poverty Level: Female Householder, No Husband Present: No Related Children Under 18 Years', 'PCT_SE_T113_010'), ('% Families: Income in at or Above Poverty Level', 'PCT_SE_T113_011'), ('Population Under 18 Years of Age for Whom Poverty Status Is Determined:', 'SE_T114_001'), ('Population Under 18 Years of Age for Whom Poverty Status Is Determined: Living in Poverty', 'SE_T114_002'), ('Population Under 18 Years of Age for Whom Poverty Status Is Determined: at or Above Poverty Level', 'SE_T114_003'), ('% Population Under 18 Years of Age for Whom Poverty Status Is Determined: Living in Poverty', 'PCT_SE_T114_002'), ('% Population Under 18 Years of Age for Whom Poverty Status Is Determined: at or Above Poverty Level', 'PCT_SE_T114_003'), ('Total:', 'SE_T130_001'), ('Total: Same House 1 Year Ago', 'SE_T130_002'), ('Total: Moved Within Same County', 'SE_T130_003'), ('Total: Moved From Different County Within Same State', 'SE_T130_004'), ('Total: Moved From Different State', 'SE_T130_005'), ('Total: Moved From Abroad', 'SE_T130_006'), ('% Total: Same House 1 Year Ago', 'PCT_SE_T130_002'), ('% Total: Moved Within Same County', 'PCT_SE_T130_003'), ('% Total: Moved From Different County Within Same State', 'PCT_SE_T130_004'), ('% Total: Moved From Different State', 'PCT_SE_T130_005'), ('% Total: Moved From Abroad', 'PCT_SE_T130_006'), ('Total Population:', 'SE_T133_001'), ('Total Population: Native Born', 'SE_T133_002'), ('Total Population: Foreign Born', 'SE_T133_003'), ('Total Population: Foreign Born: Naturalized Citizen', 'SE_T133_004'), ('Total Population: Foreign Born: Not a Citizen', 'SE_T133_005'), ('% Total Population: Native Born', 'PCT_SE_T133_002'), ('% Total Population: Foreign Born', 'PCT_SE_T133_003'), ('% Total Population: Foreign Born: Naturalized Citizen', 'PCT_SE_T133_004'), ('% Total Population: Foreign Born: Not a Citizen', 'PCT_SE_T133_005'), ('% Total Population: Under 18 Years', 'PCT_SE_T009_002'), ('% Total Population: 18 to 34 Years', 'PCT_SE_T009_003'), ('% Total Population: 35 to 64 Years', 'PCT_SE_T009_004'), ('% Total Population: 65 and Over', 'PCT_SE_T009_005'), ('Population 15 Years and Over:', 'SE_T022_001'), ('% Population 15 Years and Over: Never Married', 'PCT_SE_T022_002'), ('% Population 15 Years and Over: Now Married (Not Including Separated)', 'PCT_SE_T022_003'), ('% Population 15 Years and Over: Separated', 'PCT_SE_T022_004'), ('% Population 15 Years and Over: Widowed', 'PCT_SE_T022_005'), ('% Population 15 Years and Over: Divorced', 'PCT_SE_T022_006'), ('% Population 3 Years and Over Enrolled in School: Public School', 'PCT_SE_T029_002'), ('% Population 3 Years and Over Enrolled in School: Public School: Pre-School', 'PCT_SE_T029_003'), ('% Population 3 Years and Over Enrolled in School: Public School: K-8', 'PCT_SE_T029_004'), ('% Population 3 Years and Over Enrolled in School: Public School: 9-12', 'PCT_SE_T029_005'), ('% Population 3 Years and Over Enrolled in School: Public School: College', 'PCT_SE_T029_006'), ('% Population 3 Years and Over Enrolled in School: Private School', 'PCT_SE_T029_007'), ('% Population 3 Years and Over Enrolled in School: Private School: Pre-School', 'PCT_SE_T029_008'), ('% Population 3 Years and Over Enrolled in School: Private School: K-8', 'PCT_SE_T029_009'), ('% Population 3 Years and Over Enrolled in School: Private School: 9-12', 'PCT_SE_T029_010'), ('% Population 3 Years and Over Enrolled in School: Private School: College', 'PCT_SE_T029_011'), ('% White Alone, Not Hispanic or Latino 16 Years Old  in&nbsp; Civilian Labor Force: Employed', 'PCT_SE_T048_002'), ('% White Alone, Not Hispanic or Latino 16 Years Old  in&nbsp; Civilian Labor Force: Unemployed', 'PCT_SE_T048_003'), ('% Employed Civilian Population 16 Years and Over: Agriculture, Forestry, Fishing and Hunting, and Mining', 'PCT_SE_T049_002'), ('% Employed Civilian Population 16 Years and Over: Construction', 'PCT_SE_T049_003'), ('% Employed Civilian Population 16 Years and Over: Manufacturing', 'PCT_SE_T049_004'), ('% Employed Civilian Population 16 Years and Over: Wholesale Trade', 'PCT_SE_T049_005'), ('% Employed Civilian Population 16 Years and Over: Retail Trade', 'PCT_SE_T049_006'), ('% Employed Civilian Population 16 Years and Over: Transportation and Warehousing, and Utilities', 'PCT_SE_T049_007'), ('% Employed Civilian Population 16 Years and Over: Information', 'PCT_SE_T049_008'), ('% Employed Civilian Population 16 Years and Over: Finance and Insurance, and Real Estate and Rental  and Leasing', 'PCT_SE_T049_009'), ('% Employed Civilian Population 16 Years and Over: Professional, Scientific, and Management, and  Administrative and Waste Management Services', 'PCT_SE_T049_010'), ('% Employed Civilian Population 16 Years and Over: Educational Services, and Health Care and Social  Assistance', 'PCT_SE_T049_011'), ('% Employed Civilian Population 16 Years and Over: Arts, Entertainment, and Recreation, and  Accommodation and Food Services', 'PCT_SE_T049_012'), ('% Employed Civilian Population 16 Years and Over: Other Services, Except Public Administration', 'PCT_SE_T049_013'), ('% Employed Civilian Population 16 Years and Over: Public Administration', 'PCT_SE_T049_014'), ('% Employed Civilian Population 16 Years and Over: Unpaid Family Workers', 'PCT_SE_T053_006'), ('Median Household Income (In 2016 Inflation Adjusted Dollars): White Alone Householder', 'SE_T058_002'), ('Median Household Income (In 2016 Inflation Adjusted Dollars): Black or African American Alone Householder', 'SE_T058_003'), ('Median Household Income (In 2016 Inflation Adjusted Dollars): American Indian and Alaska Native Alone  Householder', 'SE_T058_004'), ('Median Household Income (In 2016 Inflation Adjusted Dollars): Asian Alone', 'SE_T058_005'), ('Median Household Income (In 2016 Inflation Adjusted Dollars): Native Hawaiian and Other Pacific Islander Alone  Householder', 'SE_T058_006'), ('Median Household Income (In 2016 Inflation Adjusted Dollars): Some Other Race Alone Householder', 'SE_T058_007'), ('Median Household Income (In 2016 Inflation Adjusted Dollars): Two or More Races Householder', 'SE_T058_008'), ('Median Household Income (In 2016 Inflation Adjusted Dollars): Hispanic or Latino Householder', 'SE_T058_009'), ('Median Household Income (In 2016 Inflation Adjusted Dollars): White Alone Householder, Not Hispanic or Latino', 'SE_T058_010'), ('Per Capita Income (In 2016 Inflation Adjusted Dollars)', 'SE_T083_001'), ('Median Year Structure Built', 'SE_T098_001'), ('Median Value', 'SE_T101_001'), ('% Renter-Occupied Housing Units: Less than 10 Percent', 'PCT_SE_T103_002'), ('% Renter-Occupied Housing Units: 10 to 29 Percent', 'PCT_SE_T103_003'), ('% Renter-Occupied Housing Units: 30 to 49 Percent', 'PCT_SE_T103_004'), ('% Renter-Occupied Housing Units: 50 Percent or More', 'PCT_SE_T103_005'), ('% Renter-Occupied Housing Units: Not Computed', 'PCT_SE_T103_006'), ('% Foreign-Born Population: Europe', 'PCT_SE_T139_002'), ('% Foreign-Born Population: Asia', 'PCT_SE_T139_034'), ('% Foreign-Born Population: Africa', 'PCT_SE_T139_067'), ('% Foreign-Born Population: Oceania', 'PCT_SE_T139_085'), ('% Foreign-Born Population: Americas', 'PCT_SE_T139_090'), ('Total:', 'SE_T145_001'), ('% Total: No Health Insurance Coverage', 'PCT_SE_T145_002'), ('% Total: with Health Insurance Coverage', 'PCT_SE_T145_003'), ('% Total: with Health Insurance Coverage: Public Health Coverage', 'PCT_SE_T145_004'), ('% Total: with Health Insurance Coverage: Private Health Insurance', 'PCT_SE_T145_005'), ('Occupied Housing Units', 'SE_T165_001'), ('% Occupied Housing Units: Family Households', 'PCT_SE_T165_002'),  ('% Occupied Housing Units: Family Households: Married-Couple Family', 'PCT_SE_T165_003'), ('% Occupied Housing Units: Nonfamily Households', 'PCT_SE_T165_016'), ('% Occupied Housing Units: Nonfamily Households: Householder Living Alone', 'PCT_SE_T165_017'), ('% Occupied Housing Units: With Related Children of the Householder Under 18', 'PCT_SE_T167_002'), ('% Occupied Housing Units: No Related Children of the Householder Under 18', 'PCT_SE_T167_008')

rename_dict = dict(((x,y),y) for x,y in acs_tups)

# Goal: Rename each instance of acs_words in df_charters to be the instance of acs_code corresponding in acs_tups
#chartersdf.rename(index=str, columns=rename_dict, inplace=True) # Rename columns using renaming dict
pubsdf.rename(index=str, columns=rename_dict, inplace=True) # Rename columns using renaming dict


# Save modified data to disk

print("Saving data to disk...")
#quickpickle_dump(chartersdf, "../../nowdata/backups/charters_full_2015_250_v2a_unlappedtext_counts3_geoleaid.pkl")
quickpickle_dump(pubsdf, "../../nowdata/backups/pubschools_full_2015_v2a_geoleaid.pkl")
print("...done.")

sys.exit() # Exit script (to be safe)