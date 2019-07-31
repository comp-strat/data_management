#!/usr/bin/env python
# coding: utf-8

# Author: Jaren Haber, PhD Candidate
# Institution (as of this writing): University of California, Berkeley, Dept. of Sociology
# Contributions by: Ji Shi, Statistics undergraduate, UC Berkeley (as of this writing)

# Date created: January 6, 2019
# Date last modified: January 22, 2019

# Description: Functions for calculating entity densities, years opened and closed, and entity closure rates. Built for schools, applicable to other contexts.
# GitHub repo: https://github.com/jhaber-zz/data_tools

# Import packages & functions:
import pandas
import numpy


def count_pdfs(quadruple_list, elemnum):
    """Count the number of PDFs for website in quadruple_list (i.e., a row in WEBTEXT).
    
    Args:
        quadruple_list: list of quadruples, or (URL, depth, isPDF, list_of_strings)
        elemnum: the element number of isPDF--a binary indicating whether the webpage is translated from a PDF file
    
    Returns:
        Series indicating total number of PDFs in organization (row)"""
    
    numpdfs = 0 # Initialize
    
    for page in quadruple_list:
        ispdf = page[elemnum]
        
        # Check whether elemnum is correct element number by checking contents
        if ispdf not in [0, 1, False, True, 'False', 'True']:
            print("Wrong elemnum given: not a binary indicator of PDF status. Try again!")
            return
        
        # If page comes from a PDF, add 1 to number of PDFs for school
        if ispdf in [1, True, 'True']:
            numpdfs += 1
    
    return numpdfs


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


def openclose_calc(statusdf, statusvars_list, uniqueid):
    """Determines year that each school was opened and (if applicable) closed.
    
    Args:
        DataFrame holding years opened and closed raw data,
        list of status-years to keep from DF,
        unique identifier for each entity (e.g., NCES school #).
        
    Returns:
        1-column DF holding year opened for each entity,
        1-column DF holding year closed for each entity."""
    
    # Trim input DFs to only relevant variables for finding close rates and merging, also reset index to be safe
    statusdf = statusdf[statusvars_list].reset_index(drop = True)
    
    # Define labels for open and close
    openlabel = ['YEAR_OPENED']
    closelabel = ['YEAR_CLOSED']
    
    # Get length of input DF
    length = statusdf.shape[0]
    
    cols = statusvars_list # Simplify naming for more interpretable functions
    
    # Define dictionary of status-year vars as (status col : corresponding year)
    yeardict = {'STATUS98' : 1998, 'STATUS99' : 1999, 'STATUS00' : 2000, 'STATUS01' : 2001, 
                'STATUS02' : 2002, 'STATUS03' : 2003, 'STATUS04' : 2004, 'STATUS05' : 2005,
                'STATUS06' : 2006, 'STATUS07' : 2007, 'STATUS08' : 2008, 'STATUS09' : 2009,
                'STATUS10' : 2010, 'STATUS11' : 2011, 'STATUS12' : 2012, 'STATUS13' : 2013,
                'SY_STATUS' : 2014, 'SY_STATUS15' : 2015, 'SY_STATUS16' : 2016}

    # Define helper functions for workhorse algorithm below
    def checkOpenData(statusArr, currYear):
        """Check for any open status data all the years before given year. 
        If no previous data on open status, open_year should be this year.
        If there is a number with meaningful data on openness (this excludes -1, 2, 6, 7), 
        then return False; else return True."""
        
        allNeg1 = True
        for prevStat in statusArr[:cols.index(currYear)]:
            if prevStat != -1 and prevStat != 2 and prevStat != 6 and prevStat != 7:
                allNeg1 = False
        return allNeg1

    def checkClosedAfter(statusArr, currYear):
        """Check for any closure status data all the years AFTER given year.
        If no later data on closure status, closed_year should be this year.
        If there is a 2 or 6, return True; else return False."""
        
        is2or6 = False
        for afterStat in statusArr[cols.index(currYear):]:
            if afterStat == 2 or afterStat == 6:
                is2or6 = True
        return is2or6

    def checkCloseData(statusArr):
        """Check for any closure status data ALL years for a given school.
        If there exist a 2 or 6 at any point, return True; else return False."""
        
        is2or6 = False
        for stat in statusArr:
            if stat == 2 or stat == 6:
                is2or6 = True
        return is2or6

    # Initialize the two series to store years opened and closed
    YEAR_OPENED = [numpy.NaN for i in range(length)]
    YEAR_CLOSED = [numpy.NaN for i in range(length)]

    # Main algorithm to detect year opened and closed.
    # Goes through each row for a certain interval, inclusive.
    
    for index, row in statusdf.iterrows():
        #only calculate for the input interval
        if index >= 0 and index < length:
            statusArr = [] #a list used to store all status for a year

            # Put things in statusArr
            for i in range(len(cols)):
                if numpy.isnan(row[cols[i]]):
                    statusArr += [-1] #-1 if there is a nan in certain status
                else:
                    statusArr += [int(row[cols[i]])]

            # Main logic         
            for col in cols:    
                stat = -1
                if(not numpy.isnan(row[col])):
                    stat = int(row[col])

                # If there is a 2, then close_year should be this year
                if(stat == 2):
                    YEAR_CLOSED[index] = yeardict.get(col)

                # If 1, 3, 4, 5, or 8, run checkOpenData()
                # If no previous status, open_year should be this year
                if(stat in [1, 3, 4, 5, 8]):
                    if checkOpenData(statusArr, col):
                        YEAR_OPENED[index] = yeardict.get(col)

                # If 4, run checkOpenData()
                # If no previous status, open_year should be the year before this year        
                if(stat == 4):
                    if checkOpenData(statusArr, col):
                        YEAR_OPENED[index] = yeardict.get(cols[(cols.index(col) - 1)])

                # If 8, run checkClosedAfter()
                # If no closure status data (2 or 6) after this year, year_closed should be nothing
                if(stat == 8):
                    if not checkClosedAfter(statusArr, col):
                        YEAR_CLOSED[index] = numpy.NaN

                # If 6, run checkClosedAfter()
                # If no closure status data (2 or 6) after this year, year_closed should be this year
                if(stat == 6):
                    if not checkClosedAfter(statusArr, col):
                        YEAR_CLOSED[index] = yeardict.get(col)

            # If never 2 or 6, year_closed is numpy.NaN
            # Use checkCloseData() to check this
            if not checkCloseData(statusArr):
                YEAR_CLOSED[index] = numpy.NaN
                

    # Show cases with missing data
    for index in range(length):
        if YEAR_CLOSED[index] == None or YEAR_OPENED[index] == None or YEAR_OPENED[index] == numpy.NaN:
            print("Missing data found in uniqueid | index | index in year opened array | index in year closed array:\n",
                  statusdf.loc[uniqueid, index], "|", index,"|", YEAR_OPENED[index],"|", YEAR_CLOSED[index])
            
    # Create two DataFrames from year opened/closed arrays
    dfOpened = pandas.DataFrame(YEAR_OPENED, columns = [openlabel])
    dfClosed = pandas.DataFrame(YEAR_CLOSED, columns = [closelabel])
    
    statusdf[openlabel] = dfOpened[openlabel]
    statusdf[closelabel] = dfClosed[closelabel]
    
    # Return open/close columns
    return statusdf[openlabel], statusdf[openlabel]

    '''
    # TO DO: Fix this apply() function, faster than the array-based method.
    
    def getyears_openclose(daterow):
        """Main algorithm to detect year opened and closed.
        Goes through certain interval of years, inclusive.
        
        Args:
            Row with dates of status (1 thru 8) for each year in cols (1998-2016).
            
        Returns:
            Year that entity opened, year that entity closed."""
        
        statusArr = [] # stores status values for each year

        # Initialize the two variables to store year opened and closed
        YEAR_OPENED = numpy.NaN
        YEAR_CLOSED = numpy.NaN

        # Put things in statusArr
        for i in range(len(cols)):
            if numpy.isnan(row[cols[i]]):
                statusArr += [-1] #-1 if there is a nan in certain status
            else:
                statusArr += [int(row[cols[i]])]

        # Main logic         
        for col in cols:    
            stat = -1
            if(not numpy.isnan(row[col])):
                stat = int(row[col])

            # If there is a 2, then close_year should be this year
            if(stat == 2):
                YEAR_CLOSED = yeardict.get(col)

            # If 1, 3, 4, 5, or 8, run checkOpenData()
            # If no previous status, open_year should be this year
            if(stat in [1, 3, 4, 5, 8]):
                if checkOpenData(statusArr, col):
                    YEAR_OPENED = yeardict.get(col)

            # If 4, run checkOpenData()
            # If no previous status, open_year should be the year before this year        
            if(stat == 4):
                if checkOpenData(statusArr, col):
                    YEAR_OPENED = yeardict.get(cols[(cols.index(col) - 1)])

            # If 8, run checkClosedAfter()
            # If no closure status data (2 or 6) after this year, year_closed should be nothing
            if(stat == 8):
                if not checkClosedAfter(statusArr, col):
                    YEAR_CLOSED = numpy.NaN

            # If 6, run checkClosedAfter()
            # If no closure status data (2 or 6) after this year, year_closed should be this year
            if(stat == 6):
                if not checkClosedAfter(statusArr, col):
                    YEAR_CLOSED = yeardict.get(col)

        # If never 2 or 6, year_closed is numpy.NaN
        # Use checkCloseData() to check this
        if not checkCloseData(statusArr):
            YEAR_CLOSED = numpy.NaN
            
        return YEAR_OPENED, YEAR_CLOSED
         
    statusdf['YEAR_OPENED'], statusdf['YEAR_CLOSED'] = statusdf.apply(lambda x: getyears_openclose())
    '''


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