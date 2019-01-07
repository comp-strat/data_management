#!/usr/bin/env python
# coding: utf-8

# Author: Jaren Haber, PhD Candidate
# Institution (as of this writing): University of California, Berkeley, Dept. of Sociology
# Date created: January 6, 2018
# Date last modified: January 6, 2018
# GitHub repo: https://github.com/jhaber-zz/data_tools
# Description: Functions for displaying basic DF info, storing DFs for memory efficiency, and loading a filtered DF

# Import packages & functions:
import pandas
from quickpickle import quickpickle_dump, quickpickle_load # For quickly loading & saving pickle files in Python


def check_df(DF, colname):
    """Displays basic info about a dataframe in memory.
    Input: Pandas DataFrame object
    Output: printed basic stats:    # rows and columns, 
                                    # duplicates by colname, 
                                    column names and, if missing data, the # missing cases."""
    
    # Show DF info, including # duplicates by colname
    print("# rows and cols: ", str(DF.shape))
    print("# duplicates by " + str(colname) + ": " + str(sum(DF.duplicated(subset=colname, keep='first'))))

    print("\nColumns and # missing cases (if any): ")
    for col in list(DF):
        missed = sum(DF[col].isnull())
        if missed > 0:
            print(col + ": " + str(missed) + " missing")
        else:
            print(col)


def convert_df(df, ignore_list):
    """Makes a Pandas DataFrame more memory-efficient through intelligent use of Pandas data types: 
    specifically, by storing columns with repetitive Python strings not with the object dtype for unique values 
    (entirely stored in memory) but as categoricals, which are represented by repeated integer values. This is a 
    net gain in memory when the reduced memory size of the category type outweighs the added memory cost of storing 
    one more thing. As such, this function checks the degree of redundancy for a given column before converting it."""
    
    # Remove specified columns to avoid conversion errors, those that shouldn't have their dtype converted
    # e.g., columns that are large lists of tuples, like "WEBTEXT" or "CMO_WEBTEXT", should stay as 'object' dtype
    if len(ignore_list)>0:
        ignore_df = df[ignore_list]
        df.drop(ignore_list, axis=1, inplace=True)
    
    converted_df = pandas.DataFrame() # Initialize DF for memory-efficient storage of strings (object types)
    df_obj = df.select_dtypes(include=['object']).copy() # Filter to only those columns of object data type

    # Loop through all columns that have 'object' dtype, b/c we especially want to convert these if possible:
    for col in df.columns: 
        if col in df_obj: 
            num_unique_values = len(df_obj[col].unique())
            num_total_values = len(df_obj[col])
            if (num_unique_values / num_total_values) < 0.5: # Only convert data types if at least half of values are duplicates
                converted_df.loc[:,col] = df[col].astype('category') # Store these columns as dtype "category"
            else: 
                converted_df.loc[:,col] = df[col]
        else:    
            converted_df.loc[:,col] = df[col]
                      
    # Downcast dtype to reduce memory drain
    converted_df.select_dtypes(include=['float']).apply(pandas.to_numeric,downcast='float')
    converted_df.select_dtypes(include=['int']).apply(pandas.to_numeric,downcast='signed')
    
    # Reintroduce ignored columns into resulting DF
    if len(ignore_list)>0:
        for col in ignore_list:
            converted_df[col] = ignore_df[col]
    
    return converted_df


def load_filtered_df(dfpath, keepcols):
    """Quickly loads a Pandas DataFrame from file (either .csv or .pkl format), 
    keeps only those variables in keepvars (if not an empty list), and makes the DF memory-efficient.
    Input: file path to DataFrame (.csv or .pkl), list of variables to keep from said DF (or empty list, to keep all cols)
    Output: DF with reduced variables and with memory-efficient dtypes."""
    
    if len(keepcols)>0:
        if dfpath.endswith(".csv"):
            newdf = pandas.read_csv(dfpath, usecols=keepcols, low_memory=False)
        elif dfpath.endswith(".pkl"):
            newdf = quickpickle_load(dfpath)
            newdf = newdf[keepcols]
            
    else:
        if dfpath.endswith(".csv"):
            newdf = pandas.read_csv(dfpath, low_memory=False)
        elif dfpath.endswith(".pkl"):
            newdf = quickpickle_load(dfpath)
    
    if "WEBTEXT" in list(newdf) and "CMO_WEBTEXT" in list(newdf):
        newdf = convert_df(newdf, ["WEBTEXT", "CMO_WEBTEXT"])
    elif "WEBTEXT" in list(newdf) and "CMO_WEBTEXT" not in list(newdf):
        newdf = convert_df(newdf, ["WEBTEXT"])
    elif "WEBTEXT" not in list(newdf) and "CMO_WEBTEXT" in list(newdf):
        newdf = convert_df(newdf, ["CMO_WEBTEXT"])
    else:
        newdf = convert_df(newdf, [])
    
    if "NCESSCH" in list(newdf):
        newdf["NCESSCH"] = newdf["NCESSCH"].astype(float)
        check_df(newdf, "NCESSCH")
    
    return newdf