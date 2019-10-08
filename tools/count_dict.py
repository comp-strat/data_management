#!/usr/bin/env python
# coding: utf-8

# # Dictionary Counting Script

# - Authors: Brian Yimin Lei, Jaren Haber
# - Institution: University of California, Berkeley
# - Date created: Spring 2018
# - Date last modified: September 27, 2019
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
from itertools import repeat

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
    
    Args: 
    dictpath: path to folder containing dictionaries
    dictnames: dictionary filenames (without extensions)
    fileext: file extension for all files (must all be the same)
    
    Returns:
    dict_list: List of lists, where each list contains all terms for a dictionary
    """
    
    dict_list = []
    for name in dictnames:
        with open(dictpath+name+fileext) as f: 
            new_dict = f.read().splitlines()
            dict_list.append(list(set(new_dict)))
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
    Completes dictionary to include entries with slashes ("/"), dashes ("-"), and underscores ("_") taken out.
    
    Args:
    dict_list: list of lists of dictionary terms
    stemset: whether to stem or not
    
    Returns:
    precalc_list: Each list in this returned list corresponds to a dictionary and contains five lists: 
        key_words: each term both with AND without any punctuation ( -/_), represented as a list;
        large_words: a list of large words (>2 words long); 
        large_lengths: a list of their lengths; and
        large_first_words: list of first words of any large words in dict"""
    
    precalc_list = [] # initialize list of lists holding dict info 
 
    for keywords in dict_list:  
        large_words = []
        large_lengths = []
        large_first_words = []
        key_words = []
        
        # Separate keywords by length:
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


def create_cols(df, dict_list, dict_names, text_col = "WEBTEXT", stemset = 0, mp = True):
    
    """Creates count, ratio, and strength columns for each dictionary file with [FILE_NAME]_COUNT/RATIO/STR as column name. 
    To calculate strengths in context with occasional zero-hit count, 1 is added to all counts. 
    Also, just in case, sets strengths with zero hit count to -6 (this shouldn't happen).
    
    Args:
    df: DataFrame with text data, each of which is a list of full-text pages (not necessarily preprocessed)
    dict_list: list of lists of dictionary terms, ordered same as dict_names
    dict_names: names of dictionaries on file (list or list of lists), ordered same as dict_list
    text_col: name column in df with text data (default "WEBTEXT")
    clean_text: whether to quickly clean (using regex) the web pages before searching by removing non-words and underscores
    stemset: 1 for stemming before matching, 0 for no stemming
    mp: whether this function will be used in parallel via multiprocessing
    
    Returns:
    df: modified dataframe, now contains NUMWORDS, D_COUNT, D_RATIO, and D_STR (where D is taken from dict filenames)
    """
    
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
            print('Time Elapsed:{:f}, Percent Complete:{:f}'.format(end - start,i*100/len(df)))
            
    # Store and return results:
    df['NUMWORDS'] = np.array(num_words)
    for i, name in enumerate(dict_names):
        df['{}_COUNT'.format(name.upper())] = np.array(counts[i])
        df['{}_RATIO'.format(name.upper())] = np.array(counts[i])/(np.array(num_words) - np.array(res_list[i]))
        df['{}_STR'.format(name.upper())] = np.log10(np.array([(row + (1)) for row in counts[i]]))/np.log10(np.array(num_words) - np.array(res_list[i]))
    df.replace([np.inf, -np.inf], -6, inplace = True)
    return df


def count_words(df, dict_list, dict_names, text_col = "WEBTEXT", cleantext = True, stemset = 0, mp = True):
    
    """Counts words in lists of terms (dictionaries) in a corpus.
  
    Args:
    df: DataFrame with text data, each of which is a list of full-text pages (not necessarily preprocessed)
    dict_list: list of lists of dictionary terms, ordered same as dict_names
    dict_names: names of dictionaries on file (list or list of lists), ordered same as dict_list
    text_col: name column in df with text data (default "WEBTEXT")
    clean_text: whether to quickly clean (using regex) the web pages before searching by removing non-words and underscores
    stemset: 1 for stemming before matching, 0 for no stemming
    mp: whether this function will be used in parallel via multiprocessing
    
    Returns:
    word_counts: list of lists, individual term count nested in dictionaries
    """
    
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


def collect_counts(word_counts, dict_list, dict_names, mp=True):
    """Collects counts per term per dictionary--combining across entities--into DataFrames.
    
    Args:
    word_counts: individual term counts nested in dictionaries (list of lists)--possibly nested in chunks if mp was used
    local_dicts: list of local dictionaries formatted as last of lists of terms--or if singular, just a list of terms
    local_names: names of local dictionaries (list or list of lists)
    dict_path: file path to folder containing dictionaries
    dict_names: names of dictionaries on file (list or list of lists)
    file_ext: file extension for dictionary files (probably .txt)
    mp: whether multiprocessing was used to collect word_counts (affects nesting)
    
    Returns: 
    dfs: list of DataFrames, one each showing counts for each dictionary counted, sorted by frequency
    """
    
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

    while i < len(dict_names): # repeat as many times as there are dictionaries (using names to count)
        wordlist, countlist = zipper.__next__() # grab pair of zipped lists
        total = np.sum(np.array(countlist), 0) # add up all chunks of counts (from multiprocessing) to get overall counts

        data = []; j = 0 # initialize list of word : counts and 2nd counter
        while j < len(dict_list[i]):
            data.append([wordlist[j], total[j]])
            j += 1

        dfs[i] = pd.DataFrame(data, columns=["TERM", "FREQUENCY"])
        dfs[i].sort_values(by = "FREQUENCY", ascending = False, inplace = True)
        i += 1

    return dfs


def count_master(df, dict_path, dict_names, file_ext, local_dicts, local_names, text_col = "WEBTEXT", termsonly = True, mp = True):
    """Performs dictionary counting at both entity and dictionary levels, and collects counts.
    Dictionaries (lists of terms) must be ordered same as list of names of dictionaries. 
    Can pass in dictionaries directly (local_dicts) or via file paths. 
    Runs in parallel. Each process errors out when finished with "divide by zero" and "invalid value". THIS IS NORMAL.

    Args:
    df: DataFrame with text data, each of which is a list of full-text pages (not necessarily preprocessed)
    dict_path: file path to folder containing dictionaries
    dict_names: names of dictionaries on file (list or list of lists)
    file_ext: file extension for dictionary files (probably .txt)
    local_dicts: list of local dictionaries formatted as last of lists of terms
    local_names: names of local dictionaries (list or list of lists)
    text_col: name column in df with text data (default "WEBTEXT")
    termsonly (binary): whether to only capture counts for terms (rather than entities)
    mp (binary): whether or not function will be run in parallel

    Returns:
    df_new: dataframe containing NUMWORDS, D_COUNT, D_RATIO, and D_STR (where D is taken from dict filenames)
    countsdfs: list of DataFrames, one each showing counts for each dictionary counted, sorted by frequency
    """

    file_dicts_number = len(dict_names); local_dicts_number = len(local_names) # Makes comparisons faster
    if file_dicts_number>0: # If there are dicts to be loaded from file...
        dict_list = load_dict(dict_path, dict_names, file_ext) # Load dictionaries from file
    if file_dicts_number>0 and local_dicts_number>0: # If there are dicts on file AND local dicts...
        dict_list += local_dicts # full list of dictionary names
        dict_names += local_names # full list of dictionaries
    else: # If there are only local dicts...
        dict_list = local_dicts
        dict_names = local_names
    print("Loaded dictionaries: " + str(file_dicts_number) + " from file and " + str(local_dicts_number) + " locally.")
        
    # Replace underscores, dashes, slashes, & spaces in any keywords with (1) nothing or with (2) space, adding these to dictionary:
    #new_words = [[] for _ in dict_list] # initialize new list of lists to mirror dict_list
    for d, dic in enumerate(dict_list):
        dict_list[d].extend([re.sub('/+|-+|_+', ' ', entry) for entry in dic]) # replace chars with spaces
        dict_list[d].extend([re.sub(' +|/+|-+|_+', '', entry) for entry in dic]) # replace chars and spaces with nothing
        #dic.extend(new_words)
        #dict_list[d] = list(set(new_words)) 
        '''# alternative code; no list comprehension:
        for entry in dic:
            new_words.append(re.sub(' +|/+|-+|_+', '', entry))
            new_words.append(re.sub(' +|/+|-+|_+', ' ', entry))
        '''
    dict_list = [list(set(dic)) for dic in dict_list] # eliminate duplicates
    
    # If specified, run without multiprocessing = MUCH SLOWER (no stemming by default):
    if not mp:
        
        if termsonly: # Count only words per dictionary (not per entity)
            print("Counting dictionaries per term for corpus...")
            wordcounts = count_words(tqdm(df), dict_list, dict_names, mp = False)
            print("Collecting counts per term per dictionary...")
            countsdfs = collect_counts(tqdm(wordcounts), dict_list, dict_names, mp = False)
            
            print("Finished. Returning results.")
            #for d, dic in enumerate(dict_names):
            #    print("TERM COUNTS FOR " + str(dict_names[d].upper()) + " DICTIONARY:\n")
            #    print(countsdfs[d])

            return countsdfs
        
        print("Counting dictionary totals per entity for corpus...")
        df_new = create_cols(tqdm(df), dict_list, dict_names, stemset = 0, mp = False)
        print("Counting dictionaries per term for corpus...")
        wordcounts = count_words(tqdm(df), dict_list, dict_names, mp = False)
        print("Collecting counts per term per dictionary...")
        countsdfs = collect_counts(tqdm(wordcounts), dict_list, dict_names, mp = False)
        
        print("Finished counting. Returning results.")
        #for d, dic in enumerate(dict_names):
        #    print("TERM COUNTS FOR " + str(dict_names[d].upper()) + " DICTIONARY:\n")
        #    print(countsdfs[d])
            
        return df_new, countsdfs

    # Or, run in parallel using default settings (this means with multiprocessing and no stemming)
    with multiprocessing.Pool(processes = multiprocessing.cpu_count() - 1) as pool:
        c = 1 # Define chunk size for CPU task allocation
        
        if termsonly: # Count only words per DICTIONARY (entity totals):
            print("Counting dictionaries per term for corpus...")
            wordcounts = pool.map(partial(count_words, dict_list = dict_list, dict_names = dict_names, mp = True), tqdm([df[i*c:c*(i+1)] for i in range(round(len(df)/c)+1)]))
            print("Multiprocessing complete. Collecting counts per term per dictionary...")
            countsdfs = collect_counts(tqdm(wordcounts), dict_list = dict_list, dict_names = dict_names, mp = True)
            
            print("Finished counting. Returning results.")
            #for d, dic in enumerate(dict_names):
            #    print("TERM COUNTS FOR " + str(dict_names[d].upper()) + " DICTIONARY:\n")
            #    print(countsdfs[d])
                
            return countsdfs
        
        # Count words per ENTITY (dictionary totals):
        print("Counting dictionary totals per entity for corpus...")
        results = pool.map(partial(create_cols, dict_list = dict_list, dict_names = dict_names, mp = True), tqdm([df[i*c:c*(i+1)] for i in range(round(len(df)/c)+1)])) 
        df_new = pd.concat(results)
        # Count words per DICTIONARY (entity totals):
        #print("Counting dictionaries per term for corpus...")
        #wordcounts = pool.map(partial(count_words, dict_list = dict_list, dict_names = dict_names, mp = True), tqdm([df[30*i:i*30+30] for i in range(round(len(df)/30)+1)]))

    # Collect counts from  multiprocessing into DFs:
    #print("Multiprocessing complete. Collecting counts per term per dictionary...")
    #countsdfs = collect_counts(tqdm(wordcounts), dict_list = dict_list, dict_names = dict_names, mp = True)
    
    print("Finished counting. Returning results.")
    for d, dic in enumerate(dict_names):
        print("TERM COUNTS FOR " + str(dict_names[d].upper()) + " DICTIONARY:\n")
        print(countsdfs[d])

    return df_new, countsdfs


# ## Count dictionaries across documents

# Set parameters:
#charter_path = '../misc_data/charters_2015.pkl' # path to charter school data file
#dict_path = '/home/jovyan/work/text_analysis/dictionary_methods/dicts/'
#dict_names = ['inquiry30', 'discipline30'] # enter list of names of txt files holding dict
#file_ext = '.txt'
#local_dicts = [] # list of local dictionaries formatted as last of lists of terms--or if singular, just a list of terms
#local_names = [] # names of local dictionaries (list or list of lists)
#names = dict_names + local_names # full list of dictionaries (might be either or both file-based or local)

#df_new, countsdfs = count_master() # execute master function


# ## Save results

# Drop WEBTEXT to keep file size small:
#df_new.drop(columns = "WEBTEXT", inplace=True)

# Save per entity, dictionary total counts:
#df_new.to_csv('../../charter_data/dict_counts/30counts_2015_250_v2a.csv')

# Save per dictionary, entity total counts:
#for i, df in enumerate(countsdfs):
#    df.to_csv('../../charter_data/dict_counts/{}_terms_counts_2015_250_v2a.csv'.format(names[i]))