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
        Specific (and detailed) DataFrame to merge to, 
        more general DF (for context info, e.g. all public schools), 
        column to group by (e.g., school district LEAID), 
        unique identifier for each entity (e.g., NCES school #),
        variable to filter by (e.g., charter identifier).
    Returns: 
        Density columns ready to be added to specific DataFrame."""

    # Use filtering variable to filter to more specific level (i.e., to only charter schools)
    filtered_df = fulldf[fulldf[filtervar] == 1]
    
    # Keep only relevant variables from fulldf for finding density
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


def closerate_calc(somedf, openclosedf, groupvar, openvar, closevar, uniqueid, startbound, endbound):
    """Calculates closure rate for entities that share a given grouping variable, for each year from startbound to endbound. 
    Useful for calculating closure rates (# of entities closed between year-1 and year) of public schools. 
    
    Args:
        Specific (and detailed) DataFrame to merge to,
        DataFrame listing all entities (e.g., all public schools) with integer columns for year opened and year opened (between two years indicated),
        column to group by (e.g., school district "GEO_LEAID"), 
        column indicating when entity opened  (e.g., "YEAR_OPENED"), 
        column indicating when entity closed, if at all (e.g., "YEAR_CLOSED"),
        unique identifier for each entity (e.g., NCES school #),
        first year to find closure rates (e.g., 1999),
        last year to find closure rates (e.g., 2016). 
        
    Returns:
        DataFrame containing: 
            groupvar, 
            close rate within groupvar each year (# entities closed between year-1 and year)."""
    
    # Trim input DFs to only relevant variables for finding close rates and merging
    somedf = somedf[[groupvar, uniqueid]]
    openclosedf = openclosedf[[groupvar, uniqueid, openvar, closevar]]
    
    #Create two mappings
    #  1. groupvar - list of number of schools opened in each year (don't return this)
    #  2. groupvar - list of number of schools closed in each year

    numSch_map = {} #{groupvar: [year99opened, year00opened, ..., year2016opened]}
    closed_map = {} #{groupvar: [year99closed, year00closed,..., year16closed]}
    span = (int(endbound) - int(startbound)) + 1
    
    for index, row in openclosedf.iterrows():
        thisid = row[groupvar]
        open_year = row[openvar] if not numpy.isnan(row[openvar]) else 0  #let year be 0 if not found
        close_year = row[closevar] if not numpy.isnan(row[closevar]) else 0
        if numpy.isnan(thisid):
            continue

        if thisid in numSch_map:
            for i in range(0, span):
                #if i is in the range of open years for some school, add it into the corresponding map
                if open_year <= (startbound + i) and (close_year == 0 or close_year >= (startbound + i)):
                    numSch_map[thisid][i] += 1
                if close_year == (startbound + i):
                    closed_map[thisid][i] += 1
        else:
            numSch_map[thisid] = []
            closed_map[thisid] = []
            for i in range(0, span):
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
    
    # Turn the groupvar from index to a new column
    df_closeSchool[groupvar] = df_closeSchool.index
    df_closeRate[groupvar] = df_closeRate.index
    
    # Let the groupvar column appear at the front
    mid = df_closeRate[groupvar]
    df_closeRate.drop(labels=[groupvar], axis=1,inplace = True)
    df_closeRate.insert(0, groupvar, mid)
    
    # Create closerate columns matched to original DF
    somedf = pandas.merge(somedf, df_closeRate, how='outer', on=[groupvar])
    
    print("Returning DF of closure rates for each year from " + str(startbound) + " to " + str(endbound) + ", indexed by digits from 0 to " + str(int(endbound) - int(startbound)) + "...")
          
    return somedf