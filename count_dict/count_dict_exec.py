#!/usr/bin/env python
# coding: utf-8

# # Dictionary Counting Script

# - Authors: Jaren Haber, Brian Yimin Lei
# - Institution: University of California, Berkeley
# - Date created: Spring 2018
# - Date last modified: October 21, 2019
# - Description: Finds the number of occurences of a dictionary phrase in the webtext of a school. Creates column for number of webtext words, ratio of hits, and hit strength(log of ratio). Also has a function to count words and display their frequencies. Has multiprocessing built in.


# ## Import packages

import pandas as pd
import re
import numpy as np
import time
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer = PorterStemmer()
stem = stemmer.stem # stemmer function
import multiprocessing
from functools import partial

# Define file paths
root = '/vol_b/data/'
wem_path = root + 'wem4themes/data/wem_model_300dims.bin' # path to WEM model
charter_path = root + 'misc_data/charters_2015.pkl' # path to charter school data file
dict_path = root + 'text_analysis/dictionary_methods/dicts/' # path to dictionary files (may not be used here)

# For loading functions from files in data_tools directory:
import sys; sys.path.insert(0, root + "data_management/tools/")

# For displaying basic DF info, storing DFs for memory efficiency, and loading a filtered DF:
from df_tools import check_df, convert_df, load_filtered_df, replace_df_nulls

# For quickly loading & saving pickle files in Python:
from quickpickle import quickpickle_dump, quickpickle_load 

# For saving and loading text lists to/from file:
from textlist_file import write_list, load_list 

# For calculating densities, years opened and closed, and school closure rates:
from df_calc import count_pdfs, density_calc, openclose_calc, closerate_calc

# Core functions for counting dictionaries/word frequencies in corpus:
from count_dict import load_dict, Page, dict_precalc, dict_count, create_cols, count_words, collect_counts, count_master

# For counting term frequencies, load text corpus:
print("Loading text corpus for term counting...")
df = load_filtered_df(charter_path, ["WEBTEXT", "NCESSCH"])
df['WEBTEXT']=df['WEBTEXT'].fillna('') # turn nan to empty iterable for future convenience


# ## Count dictionaries across documents

# Set dict counting parameters:
dict_names = ['inquiry5', 'inquiry20_new', 'inquiry50_new', 'inquiry49_new_handsoff'] # enter list of names of txt files holding dict
file_ext = '.txt'
local_dicts = [] # list of local dictionaries formatted as last of lists of terms--or if singular, just a list of terms
local_names = [] # names of local dictionaries (list or list of lists)

# execute master counting function
df_new = count_master(df, dict_path = dict_path, dict_names = dict_names, file_ext = '.txt', 
                                 local_dicts = local_dicts, local_names = local_names, termsonly = False)


# ## Save results

# Drop WEBTEXT to keep file size small:
df_new.drop(columns = "WEBTEXT", inplace=True)

# Save per entity, dictionary total counts:
df_new.to_csv(root + 'charter_data/dict_counts/inquiry_counts_new_2015_250_v2a.csv', index = False)

# Save per dictionary, entity total counts:
#for i, df in enumerate(countsdfs):
#    df.to_csv('../../charter_data/dict_counts/{}_terms_counts_2015_250_v2a.csv'.format(names[i]))