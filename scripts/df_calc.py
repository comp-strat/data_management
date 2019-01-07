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