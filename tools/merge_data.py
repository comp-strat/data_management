# Functions to merge data sources

# Author: Ji Shi
# Institution: University of California, Berkeley

# Project: Charter school identities
# Date: Fall 2018


import pandas as pd


def merge_dfs(file_tuples, cols, method = 'outer'):
    """
    Given a list of tuples of files, merge the two files in the tuple.
    
    For example: 
        Given ((A, B), (C, D), (E, F))
        Should return (A + B, C + D, E + F) where "+" means merge.
        
    Args: 
        file_tuples(list of tuples of strs): a list of tuples of files.
        cols(list of strs): a list of the keys that used as the identifiers for each tuples to merge.
        method(str): default is "outer", can be "left", "right" or "inner" depends on which sides you want to preserve.
    
    Returns:
        mergedDFs(list of dfs): a list of merged pandas DataFrames.
    """
    mergedDFs = []
    
    for i in range(0, len(file_tuples)):
        left = pd.read_csv(file_tuples[i][0], encoding = 'latin-1')  # the params of read_csv can change depend on your purpose.
        right = pd.read_csv(file_tuples[i][1], encoding = 'latin-1')
        df = pd.merge(left, right, how = method, on = cols[i])
        mergedDFs.append(df)

    return mergedDFs


def merge_to_one(source, files, col, method = 'left'):
    """
    Given a list of files and a main file, merge all the files in the list to the main file.
    
    For example:
       Given A and (B, C, D, E)
       Should return mergedDF = (((A + B) + C) + D) + E where "+" means merge.
       
    Args:
       source(str): the main file which you want to keep all its rows
       files(list of strs): a list of files that you want to merge to source
       col(str): the key that used as the identifier to merge.
       method(str): in this case "left" is recommanded because all other files are merged to source which is the left file.
       
    Returns:
       mergedDF(df): the merged pandas DataFrame.
    """
    
    mergedDF = pd.read_csv(source, encoding = 'latin-1')
    for f in files:
        this_file = pd.read_csv(f, encoding = 'latin-1') # the params of read_csv can change depend on your purpose.
        mergedDF = pd.merge(mergedDF, this_file, how = method, on = col)
        
    return mergedDF
    