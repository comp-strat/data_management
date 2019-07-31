#!/usr/bin/env python
# coding: utf-8

# Author: Jaren Haber, PhD Candidate
# Institution (as of this writing): University of California, Berkeley, Dept. of Sociology
# Date created: January 6, 2018
# Date last modified: January 6, 2018
# GitHub repo: https://github.com/jhaber-zz/data_tools
# Description: Functions for working with text list files

# Import packages & functions:
import pandas as pd


def write_list(file_path, textlist):
    """Writes textlist to file_path. Useful for recording output of parse_school().
    Input: Path to file, list of strings
    Output: Nothing (saved to disk)"""
    
    with open(file_path, 'w') as file_handler:
        
        for elem in textlist:
            file_handler.write("{}\n".format(elem))
    
    return    


def load_list(file_path):
    """Loads list into memory. Must be assigned to object.
    Input: Path to file
    Output: List object"""
    
    textlist = []
    with open(file_path) as file_handler:
        line = file_handler.readline()
        while line:
            textlist.append(line)
            line = file_handler.readline()
    return textlist