from random import randint
import pandas as pd
import ast
import numpy as np
import torch

#Import general packages
import imp, importlib # For working with modules
import nltk # for natural language processing tools
import pandas as pd # for working with dataframes
#from pandas.core.groupby.groupby import PanelGroupBy # For debugging
import numpy as np # for working with numbers
import pickle # For working with .pkl files
from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
import sys # For terminal tricks
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import timeit # For counting time taken for a process
import datetime # For working with dates & times

# Import packages for cleaning, tokenizing, and stemming text
import re # For parsing text
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words (it just cuts off the ends)
from nltk.stem.porter import PorterStemmer # approximate but effective (and common) method of normalizing words: stems words by implementing a hierarchy of linguistic rules that transform or cut off word endings
stem = PorterStemmer().stem # Makes stemming more accessible
from nltk.corpus import stopwords # for eliminating stop words
import gensim # For word embedding models
from gensim.models.phrases import Phrases # Makes word2vec more robust: Looks not just at  To look for multi-word phrases within word2vec

# Import packages for multiprocessing
import os # For navigation
numcpus = len(os.sched_getaffinity(0)) # Detect and assign number of available CPUs
from multiprocessing import Pool # key function for multiprocessing, to increase processing speed
pool = Pool(processes=numcpus) # Pre-load number of CPUs into pool function
import Cython # For parallelizing word2vec
mpdo = False # Set to 'True' if using multiprocessing--faster for creating words by sentence file, but more complicated
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

# For loading functions from files in data_tools directory:
import sys; sys.path.insert(0, "../../../../data_tools/")
from clean_text import clean_sentence, stopwords_make, punctstr_make, unicode_make
import clean_text

# ## Create lists of stopwords, punctuation, and unicode characters
stop_words_list = stopwords_make() # Define old vocab file path if you want to remove first, dirty elements
unicode_list = unicode_make()
punctstr = punctstr_make()

# Load model
from models import InferSent
model_version = 1
MODEL_PATH = "../encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'glove.840B.300d.txt'
model.set_w2v_path(W2V_PATH)

folder_prefix = '/vol_b/data/'
#df = pd.read_pickle(folder_prefix + 'nowdata/charters_2015.pkl') #original
df = pd.read_pickle(folder_prefix + 'nowdata/parsing/filtered_10.pkl') #10 pages limit


school_sentslist = []

def preprocess_wem(tuplist): # inputs were formerly: (tuplist, start, limit)
    
    '''This function cleans and tokenizes sentences, removing punctuation and numbers and making words into lower-case stems.
    Inputs: list of four-element tuples, the last element of which holds the long string of text we care about;
        an integer limit (bypassed when set to -1) indicating the DF row index on which to stop the function (for testing purposes),
        and similarly, an integer start (bypassed when set to -1) indicating the DF row index on which to start the function (for testing purposes).
    This function loops over five nested levels, which from high to low are: row, tuple, chunk, sentence, word.
    Note: This approach maintains accurate semantic distances by keeping stopwords.'''
        
    global mpdo # Check if we're doing multiprocessing. If so, then mpdo=True
    global sents_combined # Grants access to variable holding a list of lists of words, where each list of words represents a sentence in its original order (only relevant for this function if we're not using multiprocessing)
    global pcount # Grants access to preprocessing counter
    
    known_pages = set() # Initialize list of known pages for a school
    sents_combined = [] # Initialize list of all school's sentences

    if type(tuplist)==float:
        return # Can't iterate over floats, so exit
    
    #print('Parsing school #' + str(pcount)) # Print number of school being parsed

    for tup in tuplist: # Iterate over tuples in tuplist (list of tuples)
        if tup[3] in known_pages or tup=='': # Could use hashing to speed up comparison: hashlib.sha224(tup[3].encode()).hexdigest()
            continue # Skip this page if exactly the same as a previous page on this school's website

        for chunk in tup[3].split('\n'): 
            for sent in sent_tokenize(chunk): # Tokenize chunk by sentences (in case >1 sentence in chunk)
                #sent = clean_sentence(sent, fast=True) # Clean and tokenize sentence
                sent = clean_sentence(sent)
                if ((sent == []) or (len(sent) == 0)): # If sentence is empty, continue to next sentence without appending
                    continue
                
                
                # TO DO: Chunk this by school, not just sentence
                # TO DO: Now that sentences are parsed and cleaned by spaces, 
                # recombine and then parse more accurately using spacy word tokenizer
                
                # Save preprocessing sentence to object (if not multiprocessing)
                #sents_combined.append(sent) # add sent to object #if nested works
                sents_combined.extend(sent) # if nested version doesnt work
                    
        known_pages.add(tup[3])
        
    school_sentslist.append(sents_combined) # add sent to object
    
    #pcount += 1 # Add to counter
    
    return sents_combined


def preprocess_wem2(ls):
    
    '''This function cleans and tokenizes sentences, removing punctuation and numbers and making words into lower-case stems.
    Inputs: list of strings;
    This function loops over all elements in the input list given, cleans the texts and returns one string'''
        
    global mpdo # Check if we're doing multiprocessing. If so, then mpdo=True
    global sents_combined # Grants access to variable holding a list of lists of words, where each list of words represents a sentence in its original order (only relevant for this function if we're not using multiprocessing)
    global pcount # Grants access to preprocessing counter
    
    known_pages = set() # Initialize list of known pages for a school
    sents_combined = [] # Initialize list of all school's sentences
    
    #print('Parsing school #' + str(pcount)) # Print number of school being parsed

    for s in ls: # Iterate over tuples in tuplist (list of tuples)
        for chunk in s.split('\n'): 
            for sent in sent_tokenize(chunk): # Tokenize chunk by sentences (in case >1 sentence in chunk)
                #sent = clean_sentence(sent, fast=True) # Clean and tokenize sentence
                sent = clean_sentence(sent)
                if ((sent == []) or (len(sent) == 0)): # If sentence is empty, continue to next sentence without appending
                    continue
                
                
                # TO DO: Chunk this by school, not just sentence
                # TO DO: Now that sentences are parsed and cleaned by spaces, 
                # recombine and then parse more accurately using spacy word tokenizer
                
                # Save preprocessing sentence to object (if not multiprocessing)
                #sents_combined.append(sent) # add sent to object #if nested works
                sents_combined.extend(sent) # if nested version doesnt work
    school_sentslist.append(sents_combined) # add sent to object
    
    return sents_combined

print("Ready to start preprocessing text")

#initializing list for concatenated string for webtext per school
ls_str = []
for i in range(len(df)):
    #ls_str.append(' '.join(preprocess_wem(df['WEBTEXT'][i])))
    ls_str.append(' '.join(preprocess_wem2(df['WEBTEXT'][i])))
    if i%500 == 0:
        print("Cleaned ", i, " rows")
    
df['text_full'] = ls_str
df = df[['NCESSCH', 'text_full']]

print("Saving dataframe with new text column...")

df.to_csv("limited_pages_text.csv")

print("Saving complete!")
