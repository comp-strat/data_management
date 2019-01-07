#!/usr/bin/env python
# coding: utf-8

# Author: Jaren Haber, PhD Candidate
# Institution (as of this writing): University of California, Berkeley, Dept. of Sociology
# Date created: January 6, 2018
# Date last modified: January 6, 2018
# GitHub repo: https://github.com/jhaber-zz/data_tools
# Description: # For calculating densities (already built) and (coming soon!) school closure rates and cleaned performance variables

# Import packages & functions:
import pandas
import numpy


def density_calc(somedf, fulldf, groupvar, uniqueid, filtervar):
    """Calculates total number of entities (rows) in a given DataFrame that share a given clustering/group variable.
    Uses uniqueid to identify number of independent entities. 
    Uses a more general DF (e.g., all public schools) to calculate density of both all entities and 
    specific entities (identified using filtervar from within general DF; e.g., charter schools).
    Useful for calculating the density of charter & public schools in a given school district.
    
    Args: 
        Specific (and detailed) DataFrame, 
        more general DF (for context info, e.g. all public schools), 
        variable to group by (e.g., school district LEAID), 
        unique identifier for each entity (e.g., NCES school #),
        variable to filter by (e.g., charter identifier).
    Returns: 
        Density columns ready to be added to specific DataFrame."""

    # Use filtering variable to filter to more specific level (i.e., to only charter schools)
    filtered_df = fulldf[fulldf[filtervar] == 1]
    
    # Keep only relevant variables from fulldf for finding density (not necessary if fulldf already trimmed)
    fulldf = fulldf[[groupvar, uniqueid]]
    
    # Generate 2-element DFs grouped by groupvar, both total and filtered, 
    # identifying distinct entities using uniqueid:
    grouped_total = fulldf.groupby([groupvar])[uniqueid].count().reset_index(name="Total_num_entities")
    grouped_filtered = filtered_df.groupby([groupvar])[uniqueid].count().reset_index(name="Filtered_num_entities")
    
    # Create density columns matched to original DF
    total_density = pandas.merge(somedf, grouped_total, how='outer', on=[groupvar])["Total_num_entities"]
    filtered_density = pandas.merge(somedf, grouped_filtered, how='outer', on=[groupvar])["Filtered_num_entities"]
    
    # Could calculate density by dividing school counts by land mass of school district
    #new_frame[densityvar] = new_frame['All_school_counts']/merge_frame[("Area (Land)", "Geo_AREALAND")]
    
    return total_density, filtered_density


def closerate_calc(openclosedf, groupvar, openvar, closevar):
    """
    
    Args:
        DataFrame with variable for year opened and year opened,
        variable to group by (e.g., school district GEO_LEAID), 
        variable indicating when entity opened  (e.g., YEAR_OPENED), 
        variable indicating when entity closed, if at all (e.g., YEAR_CLOSED). 
        
    Returns:
        """
    
    #Create two mappings
    #  1. groupvar - list of number of schools opened in each year
    #  2. groupvar - list of number of schools closed in each year

    numSch_map = {} #{groupvar: [year99opened, year00opened, ..., year2016opened]}
    closed_map = {} #{groupvar: [year99closed, year00closed,..., year16closed]}
    for index, row in openclosedf.iterrows():
        thisid = row[groupvar]
        open_year = row[openvar] if not numpy.isnan(row[openvar]) else 0  #let year be 0 if not found
        close_year = row[closevar] if not numpy.isnan(row[closevar]) else 0
        if numpy.isnan(thisid):
            continue

        if thisid in numSch_map:
            for i in range(0, 18):
                #if i is in the range of open years for some school, add it into the corresponding map
                if open_year <= 1999 + i and (close_year == 0 or close_year >= 1999 + i):
                    numSch_map[thisid][i] += 1
                if close_year == 1999 + i:
                    closed_map[thisid][i] += 1
        else:
            numSch_map[thisid] = []
            closed_map[thisid] = []
            for i in range(0, 18):
                numSch_map[thisid].append(0)
                closed_map[thisid].append(0)
           
        
    # Calculating the close rate for each groupvar using the mapping we did above
    close_rate = {}
    for key in numSch_map.keys():
        for i in range(0, len(numSch_map[key])):
            denom = numSch_map[key][i]
            if key not in close_rate:
                #create a list of close rate values
                close_rate[key] = []
                if denom == 0:
                    close_rate[key].append(0)
                else:
                    close_rate[key].append(closed_map[key][i] / denom)
            else:
                if denom == 0:
                    close_rate[key].append(0)
                else:
                    close_rate[key].append(closed_map[key][i] / denom)
                   

    # Turn the close school mapping and close rate mapping into pandas dataframe
    df_closeSchool = pandas.DataFrame.from_dict(closed_map)
    df_closeRate = pandas.DataFrame.from_dict(close_rate)
    df_closeSchool = df_closeSchool.transpose()
    df_closeRate = df_closeRate.transpose()
    
    # Create two dictionary to rename the columns
    dic1 = {0:'close99', 1:'close00', 2:'close01', 3:'close02', 4:'close03', 5:'close04', 6:'close05', 7:'close06', \
           8:'close07',9:'close08', 10:'close09', 11:'close10', 12:'close11', 13:'close12', 14:'close13', 15:'close14', \
           16:'close15', 17:'close16'}
    dic2 = {0:'close_rate99', 1:'close_rate00', 2:'close_rate01', 3:'close_rate02', 4:'close_rate03', 5:'close_rate04', 6:'close_rate05', 7:'close_rate06', \
           8:'close_rate07',9:'close_rate08', 10:'close_rate09', 11:'close_rate10', 12:'close_rate11', 13:'close_rate12', 14:'close_rate13', 15:'close_rate14', \
           16:'close_rate15', 17:'close_rate16'}
    
    # Rename the two dataframe using the dictionaries created above
    df_closeSchool = df_closeSchool.rename(columns = dic1)
    df_closeRate = df_closeRate.rename(columns = dic2)
    
    # Turn the groupvar from index to a new column
    df_closeSchool[groupvar] = df_closeSchool.index
    df_closeRate[groupvar] = df_closeRate.index
    
    # Merge the closed school dataframe and the close rate dataframe
    merged_close = pandas.merge(df_closeSchool, df_closeRate, on=[groupvar])
    
    # Let the groupvar column appear at the front
    mid = merged_close[groupvar]
    merged_close.drop(labels=[groupvar], axis=1,inplace = True)
    merged_close.insert(0, groupvar, mid)
    
    
    
    return merged_close