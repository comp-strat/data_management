#!/usr/bin/env python
# coding: utf-8

# # Dictionary Counting Script

# - Authors: Brian Yimin Lei, Jaren Haber
# - Institution: University of California, Berkeley
# - Date created: Spring 2018
# - Date last modified: September 26, 2019
# - Description: Finds the number of occurences of a dictionary phrase in the webtext of a school. Creates column for number of webtext words, ratio of hits, and hit strength(log of ratio). Also has a function to count words and display their frequencies. Has multiprocessing built in.


# FIRST: Set dictionary count parameters and file paths
charter_path = '../misc_data/charters_2015.pkl' # path to charter school data file
#dict_path = '/home/jovyan/work/text_analysis/dictionary_methods/dicts/'
#dict_names = ['inquiry30', 'discipline30'] # enter list of names of txt files holding dict
#file_ext = '.txt'
#local_dicts = [] # list of local dictionaries formatted as last of lists of terms--or if singular, just a list of terms
#local_names = [] # names of local dictionaries (list or list of lists)
#names = dict_names + local_names # full list of dictionaries (might be either or both file-based or local)


# ## Import packages

import pandas as pd
import re
import numpy as np
import time
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer = PorterStemmer()
stem = stemmer.stem # stemmer function
import multiprocessing

# For loading functions from files in data_tools directory:
import sys; sys.path.insert(0, "../../data_management/tools/")

# For displaying basic DF info, storing DFs for memory efficiency, and loading a filtered DF:
from df_tools import check_df, convert_df, load_filtered_df, replace_df_nulls

# For quickly loading & saving pickle files in Python:
from quickpickle import quickpickle_dump, quickpickle_load 

# For saving and loading text lists to/from file:
from textlist_file import write_list, load_list 

# For calculating densities, years opened and closed, and school closure rates:
from df_calc import count_pdfs, density_calc, openclose_calc, closerate_calc


# ## Define functions for counting dictionaries

def load_dict(dictpath, dictnames, fileext):
    """Loads dictionaries into list.
    Completes dictionary to include entries with slashes ("/"), dashes ("-"), and underscores ("_") taken out.
    
    Args: 
    dictpath: path to folder containing dictionaries
    dictnames: dictionary filenames (without extensions)
    fileext: file extension for all files (must all be the same)
    
    Returns:
    dict_list: List of lists, where each list contains all terms for a dictionary with AND without punctuation
    """
    
    dict_list = []
    for name in dictnames:
        with open(dictpath+name+fileext) as f: 
            new_dict = f.read().splitlines()
            new_words = []
            for entry in new_dict:
                new_words.append(re.sub(' +|/+|-+|_+', '', entry))
            new_dict.extend(new_words)
            new_dict = set(new_dict)
            dict_list.append(list(new_dict))
    return dict_list


class Page:
    def __init__(self,p):
        self.url = p[0]
        self.boo = p[1]
        self.depth = p[2]
        self.text = p[3]
    def __repr__(self):
        return self.text
    def __eq__(self, other):
        if isinstance(other, Page):
            return self.text == other.text
        else:
            return False
    def __ne__(self, other):
        return (not self.__eq__(other))
    def __hash__(self):
        return hash(self.__repr__())
    
def dict_precalc(dict_list, stemset):
    """Cleans dictionaries and returns a list of lists of lists. 
    Each list in the returned list corresponds to a dictionary and contains five lists: 
    key_words: each term, represented as a list, separated by punctuation;
    large_words: a list of large words (>2 words long); 
    large_lengths: a list of their lengths; and
    large_first_words: list of first words of any large words in dict"""
    
    precalc_list = []
    for keywords in dict_list:  
        large_words = []
        large_lengths = []
        large_first_words = []
        key_words = []
        for entry in keywords:
            if stemset:
                word = [stem(x) for x in re.split('\W+|_+', entry.lower())]
            else:
                word = re.split('\W+|_+', entry.lower())
            key_words.append(word) # listified version of each keyword
            if len(word) >= 3:
                large_words.append(word) # contains only large entries(>2 word)
                large_lengths.append(len(word))
                large_first_words.append(word[0]) # first words of each large entry in dict
        precalc_list.append([key_words, large_words, large_lengths, large_first_words])
    return precalc_list

def dict_count(key_words, large_words, large_lengths, large_first_words, pages, keycount):

    """Returns the hit count with given dictionary on page set.

    pages: set of preprocessed page lists corresponding to an entry of the 'webtext' column
    
    Returns:
    counts: number of matches between pages (text) and key_words (dictionary terms)
    res_length: length of pages, adjusted to subtract extra words for long (>1 word) dictionary terms
    """
    
    counts = 0 # number of matches between text_list and custom_dict
    res_length = 0
    # Do dictionary analysis for word chunks of lengths max_entry_length down to 1
    for splitted_phrase in pages:
        for length in range(1, 3):
            if len(splitted_phrase) < length:
                continue # If text chunk is shorter than length of dict entries being matched, there are no matches.
            for i in range(len(splitted_phrase) - length + 1):
                entry = splitted_phrase[i:i+length]
                if entry in key_words:
                    counts += 1
                    res_length += length - 1
                    for i, term in enumerate(key_words):
                        if term == entry:
                            keycount[i] += 1
        indices = np.transpose(np.nonzero([np.array(splitted_phrase) == first_word for first_word in large_first_words]))
        for ind in indices:
            if ind[1] <= (len(splitted_phrase) - large_lengths[ind[0]]) and large_words[ind[0]] == splitted_phrase[ind[1] : ind[1] + large_lengths[ind[0]]]:
                counts += 1
                res_length += large_lengths[ind[0]] - 1
    return counts, res_length, keycount


def create_cols(df_charter, dict_path, dict_names, file_ext, local_dicts, local_names, stemset = 0, mp = True):
    
    """Creates count, ratio, and strength columns for each dictionary file with [FILE_NAME]_COUNT/RATIO/STR as column name. 
    Runs in parallel. Each process errors out when finished with "divide by zero" and "invalid value". THIS IS NORMAL.
    To calculate strengths in context with occasional zero-hit count, 1 is added to all counts. 
    Also, just in case, sets strengths with zero hit count to -6 (this shouldn't happen).
    Optional: define local dictionaries (passed directly into function, rather than loaded from file) to count
    
    Args:
    df_charter: DataFrame with WEBTEXT column, each of which is a list of full-text pages (not preprocessed)
    local_dicts: list of local dictionaries formatted as last of lists of terms--or if singular, just a list of terms
    local_names: names of local dictionaries (list or list of lists)
    dict_path: file path to folder containing dictionaries
    dict_names: names of dictionaries on file (list or list of lists)
    file_ext: file extension for dictionary files (probably .txt)
    stemset: 1 for stemming before matching, 0 for no stemming
    mp: whether this function will be used in parallel via multiprocessing
    
    Returns:
    df_charter: modified dataframe, now contains NUMWORDS, D_COUNT, D_RATIO, and D_STR (where D is taken from dict filenames)
    
    This function also requires one or both of these sets of global parameters to be defined:
    local_dicts: list of local dictionaries formatted as last of lists of terms--or if singular, just a list of terms
    local_names: names of local dictionaries (list or list of lists)
    AND/OR
    dict_path: file path to folder containing dictionaries
    dict_names: names of dictionaries on file (list or list of lists)
    file_ext: file extension for dictionary files (probably .txt)
    """
    
    #global local_dicts, local_names # Access to local dictionaries to count (defined in notebook) and their names
    #global dict_path, dict_names, file_ext # Access to parameters for load_dict()
    
    # Initialize and load dictionaries, both from file and locally:
    if len(local_dicts) + len(dict_names)==0:
        print("ERROR: No dictionaries detected. Stopping term counting script.")
        return
    if len(dict_names)>0:
        dict_list = load_dict(dict_path, dict_names, file_ext)
        precalc_list = dict_precalc(dict_list, stemset) # clean and sort dictionaries by length
    else:
        dict_list = local_dicts
        dict_names = local_names
    if len(dict_names)>0 and len(local_names)>0:
        dict_list += local_dicts
        dict_names += local_names
    
    # Initialize storing lists:
    counts = [[] for _ in range(len(dict_list))] # list of lists, each list a dictionary count per page for entity 
    res_list = [[] for _ in range(len(dict_list))] # list of lists, each list for lengths of pages for entity (adjusted for long dict terms)
    key_counts = [[] for _ in range(len(dict_list))] # list of lists, each list contains counts for terms for a dictionary
    num_words = [] # WARNING: hitcount/numwords will not give accurate hit ratio in case of multiple word entries. Calculated strength variables do account for this however.
    precalc_list = dict_precalc(dict_list, stemset) # clean and sort dictionaries by length
    key_counts = [[0 for _ in precalc_list[i][0]] for i, d in enumerate(key_counts)] # initialize list of term counts as all zeroes
    start = time.time()
    
    # Count occurrence of dictionary terms:
    for i, row in enumerate(df_charter['WEBTEXT'].values):
        pages = set([Page(p) for p in row])
        if stemset:
            pages = [[stem(x) for x in re.split('\W+|_', p.text)] for p in pages] # preprocess pages in same way as dictionaries should have been in above precalc function
        else:
            pages = [re.split('\W+|_', p.text) for p in pages]
        num_words.append(sum([len(p) for p in pages]))
        for j, d in enumerate(dict_list):
            c, res, key_counts[j] = dict_count(precalc_list[j][0], precalc_list[j][1], precalc_list[j][2], precalc_list[j][3], pages, key_counts[j])
            counts[j].append(c)
            res_list[j].append(res)
        if not mp and i%1000 == 0:
            end = time.time()
            print('Time Elapsed:{:f}, Percent Complete:{:f}'.format(end - start,i*100/len(df_charter)))
            
    # Store and return results:
    df_charter['NUMWORDS'] = np.array(num_words)
    for i, name in enumerate(dict_names):
        df_charter['{}_COUNT'.format(name.upper())] = np.array(counts[i])
        df_charter['{}_RATIO'.format(name.upper())] = np.array(counts[i])/(np.array(num_words) - np.array(res_list[i]))
        df_charter['{}_STR'.format(name.upper())] = np.log10(np.array([(row + (1)) for row in counts[i]]))/np.log10(np.array(num_words) - np.array(res_list[i]))
    df_charter.replace([np.inf, -np.inf], -6, inplace = True)
    return df_charter


def count_words(df, dict_path, dict_names, file_ext, local_dicts, local_names, text_col = "WEBTEXT", cleantext = True, stemset = 0, mp = True):
    # local_dicts = [], local_names = [], dict_path = "", dict_names = [], file_ext = ""
    
    """Counts words in lists of terms (dictionaries) in a corpus.
    Define dictionaries locally or by passing in dictionary file paths.
    Runs in parallel. Each process errors out when finished with "divide by zero" and "invalid value". THIS IS NORMAL.

    Args:
    df_charter: DataFrame with text data, each of which is a list of full-text pages (not necessarily preprocessed)
    local_dicts: list of local dictionaries formatted as last of lists of terms--or if singular, just a list of terms
    local_names: names of local dictionaries (list or list of lists)
    dict_path: file path to folder containing dictionaries
    dict_names: names of dictionaries on file (list or list of lists)
    file_ext: file extension for dictionary files (probably .txt)
    text_col: name column in df_charter with text data
    clean_text: whether to quickly clean (using regex) the web pages before searching by removing non-words and underscores
    stemset: 1 for stemming before matching, 0 for no stemming
    mp: whether this function will be used in parallel via multiprocessing
    
    Returns:
    word_counts: list of lists, individual term count nested in dictionaries
    
    This function also requires one or both of these sets of global parameters to be defined:
    local_dicts: list of local dictionaries formatted as last of lists of terms--or if singular, just a list of terms
    local_names: names of local dictionaries (list or list of lists)
    AND/OR
    dict_path: file path to folder containing dictionaries
    dict_names: names of dictionaries on file (list or list of lists)
    file_ext: file extension for dictionary files (probably .txt)
    """
    
    #global local_dicts, local_names # Access to local dictionaries to count (defined in notebook) and their names
    #global dict_path, dict_names, file_ext # Access to parameters for load_dict()
    
    # Initialize and load dictionaries, both from file and locally:
    if len(local_dicts) + len(dict_names)==0:
        print("ERROR: No dictionaries detected. Stopping term counting script.")
        return
    if len(dict_names)>0:
        dict_list = load_dict(dict_path, dict_names, file_ext)
    else:
        dict_list = local_dicts
        dict_names = local_names
    if len(dict_names)>0 and len(local_names)>0:
        dict_list += local_dicts
        dict_names += local_names
    
    # Initialize storing lists:
    counts = [[] for _ in range(len(dict_list))] # list of lists, each list a dictionary count per page for entity 
    res_list = [[] for _ in range(len(dict_list))] # list of lists, each list for lengths of pages for entity (adjusted for long dict terms)
    key_counts = [[] for _ in range(len(dict_list))] # list of lists, each list contains counts for terms for a dictionary
    num_words = [] # WARNING: hitcount/numwords will not give accurate hit ratio in case of multiple word entries. Calculated strength variables do account for this however.
    precalc_list = dict_precalc(dict_list, stemset) # clean and sort dictionaries by length
    key_counts = [[0 for _ in precalc_list[i][0]] for i, d in enumerate(key_counts)] # initialize list of term counts as all zeroes
    start = time.time()
    
    # Count occurrence of dictionary terms:
    for i, row in enumerate(df[text_col].values):
        pages = set([Page(p) for p in row])
        if stemset and cleantext:
            pages = [[stem(x) for x in re.split('\W+|_', p.text)] for p in pages] # preprocess pages in same way as dictionaries should have been in above precalc function
        elif cleantext:
            pages = [re.split('\W+|_', p.text) for p in pages]
        else:
            pages = [p.text.split() for p in pages]
        num_words.append(sum([len(p) for p in pages]))
        for j, d in enumerate(dict_list):
            c, res, key_counts[j] = dict_count(precalc_list[j][0], precalc_list[j][1], precalc_list[j][2], precalc_list[j][3], pages, key_counts[j])
            counts[j].append(c)
            res_list[j].append(res)
        if not mp and i%1000 == 0:
            end = time.time()
            print('Time Elapsed:{:f}, Percent Complete:{:f}'.format(end - start,i*100/len(df)))
            
    # Return results:
    return key_counts


def collect_counts(word_counts, dict_path, dict_names, file_ext, local_dicts, local_names, mp=True):
    """Collects counts per term per dictionary--combining across entities--into DataFrames.
    
    Args:
    local_dicts: list of local dictionaries formatted as last of lists of terms--or if singular, just a list of terms
    local_names: names of local dictionaries (list or list of lists)
    dict_path: file path to folder containing dictionaries
    dict_names: names of dictionaries on file (list or list of lists)
    file_ext: file extension for dictionary files (probably .txt)
    word_counts: individual term counts nested in dictionaries (list of lists)--possibly nested in chunks if mp was used
    mp: whether multiprocessing was used to collect word_counts (affects nesting)
    
    Returns: 
    dfs: list of DataFrames, one each showing counts for each dictionary counted, sorted by frequency
    """
    
    # Initialize and load dictionaries, both from file and locally:
    #global dict_path, dict_names, file_ext # Access to parameters for load_dict()
    #global local_dicts, local_names # Access to local dictionaries to count (defined in notebook) and their names

    if len(dict_names)>0:
        dict_list = load_dict(dict_path, dict_names, file_ext) # Load dictionaries from file
    else:
        dict_list = local_dicts
        dict_names = local_names
    if len(dict_names)>0 and len(local_names)>0:
        dict_list += local_dicts # full list of dictionary names
        dict_names += local_names # full list of dictionaries
    
    # Convert format from chunks of dictionaries of terms -> dictionaries of chunks of terms
    counts = [[] for _ in range(len(dict_names))]
    for i, name in enumerate(dict_names):
        if mp:
            [counts[i].append(chunk[i]) for chunk in word_counts]
        if not mp:
            [counts[i].append(word_counts[i])]


    i = 0 # first counter
    zipper = zip(dict_list, counts) # object connecting dictionaries and word counts (must be indexed the same)
    dfs = [pd.DataFrame() for _ in range(len(dict_names))] # list of DFs goes here for output

    while i < len(names): # repeat as many times as there are dictionaries (using names to count)
        print("TERM COUNTS FOR " + str(dict_names[i].upper()) + " DICTIONARY:\n")
        wordlist, countlist = zipper.__next__() # grab pair of zipped lists
        total = np.sum(np.array(countlist), 0) # add up all chunks of counts (from multiprocessing) to get overall counts

        data = []; j = 0 # initialize list of word : counts and 2nd counter
        while j < len(dict_list[i]):
            data.append([wordlist[j], total[j]])
            j += 1

        dfs[i] = pd.DataFrame(data, columns=["TERM", "FREQUENCY"])
        dfs[i].sort_values(by = "FREQUENCY", ascending = False, inplace = True)
        i += 1
        print("\n")

    return dfs


def count_master(dict_path, dict_names, file_ext, local_dicts, local_names, termsonly = True, mp = True):
    """Performs dictionary counting at both entity and dictionary levels, and collects counts.
    NOTE: This function by default uses these global variables as defined at top of count_dict.py. To override these defaults, define these manually...

    Args:
    local_dicts: list of local dictionaries formatted as last of lists of terms--or if singular, just a list of terms
    local_names: names of local dictionaries (list or list of lists)
    dict_path: file path to folder containing dictionaries
    dict_names: names of dictionaries on file (list or list of lists)
    file_ext: file extension for dictionary files (probably .txt)
    termsonly (binary): whether to only capture counts for terms (rather than entities)
    mp (binary): whether or not function will be run in parallel

    Returns:
    df_new: dataframe containing NUMWORDS, D_COUNT, D_RATIO, and D_STR (where D is taken from dict filenames)
    countsdfs: list of DataFrames, one each showing counts for each dictionary counted, sorted by frequency
    """

    #global charter_path
    #global local_dicts, local_names # Access to local dictionaries to count (defined in notebook) and their names
    #global dict_path, dict_names, file_ext # Access to parameters for load_dict(): folder path, file names, file extension (must be consistent)
    names = dict_names + local_names # full list of dictionaries (might be either or both file-based or local)
    
    df_charter1 = load_filtered_df(charter_path, ["WEBTEXT", "NCESSCH"])
    df_charter1['WEBTEXT']=df_charter1['WEBTEXT'].fillna('') # turn nan to empty iterable for future convenience

    # If specified, run without multiprocessing = MUCH SLOWER (no stemming by default):
    if not mp:
        
        if termsonly:
            wordcounts = count_words(df_charter1, dict_path, dict_names, file_ext, local_dicts, local_names, mp = False)
            countsdfs = collect_counts(wordcounts, dict_path, dict_names, file_ext, local_dicts, local_names, mp = False)
            return countsdfs
        
        df_new = create_cols(df_charter1, dict_path, dict_names, file_ext, local_dicts, local_names, stemset = 0, mp = False)
        wordcounts = count_words(df_charter1, dict_path, dict_names, file_ext, local_dicts, local_names, mp = False)
        countsdfs = collect_counts(wordcounts, dict_path, dict_names, file_ext, local_dicts, local_names, mp = False)
        return df_new, countsdfs

    # Or, run in parallel using default settings (this means with multiprocessing and no stemming)
    with multiprocessing.Pool(processes = multiprocessing.cpu_count() - 1) as pool:
        
        if termsonly:
            # Count words per DICTIONARY (entity totals):
            wordcounts = pool.map(count_words(dict_path, dict_names, file_ext, local_dicts, local_names), [df_charter1[300*i:i*300+300] for i in range(round(len(df_charter1)/300)+1)])
            countsdfs = collect_counts(wordcounts, dict_path, dict_names, file_ext, local_dicts, local_names, mp = True)
            return countsdfs
        
        # Count words per ENTITY (dictionary totals):
        results = pool.map(create_cols(dict_path, dict_names, file_ext, local_dicts, local_names), [df_charter1[300*i:i*300+300] for i in range(round(len(df_charter1)/300)+1)])
        # Count words per DICTIONARY (entity totals):
        wordcounts = pool.map(count_words(dict_path, dict_names, file_ext, local_dicts, local_names), [df_charter1[300*i:i*300+300] for i in range(round(len(df_charter1)/300)+1)])

    # Collect counts from  multiprocessing into DFs:
    df_new = pd.concat(results[0])
    countsdfs = collect_counts(wordcounts, dict_path, dict_names, file_ext, local_dicts, local_names, mp = True)

    return df_new, countsdfs


# ## Count dictionaries across documents

#df_new, countsdfs = count_master() # execute master function


# ## Save results

# Drop WEBTEXT to keep file size small:
#df_new.drop(columns = "WEBTEXT", inplace=True)

# Save per entity, dictionary total counts:
#df_new.to_csv('../../charter_data/dict_counts/30counts_2015_250_v2a.csv')

# Save per dictionary, entity total counts:
#for i, df in enumerate(countsdfs):
#    df.to_csv('../../charter_data/dict_counts/{}_terms_counts_2015_250_v2a.csv'.format(names[i]))