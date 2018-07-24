import pandas as pd
import time
import re
import numpy as np


MIN_HITCOUNT = 2 # min hit count for filtering pages
keywords = ['values', 'academics', 'academic', 'skills', 'skill', 'purpose', 'purposes',
                       'direction', 'mission', 'vision', 'visions', 'missions',
                       'ideals', 'cause', 'causes', 'curriculum', 'curricular',
                       'method', 'methods', 'pedagogy', 'pedagogical', 'pedagogies', 'approach', 'approaches', 'model', 'models', 'system', 'systems',
                       'structure', 'structures', 'philosophy', 'philosophical', 'philosophies', 'beliefs', 'believe', 'belief',
                       'principles', 'principle', 'creed', 'creeds', 'credo', 'moral', 'morals', 'morality', 'history', 'histories', 'our story',
                       'the story', 'school story', 'background', 'backgrounds', 'founding', 'founded', 'foundation', 'foundations', 'foundational',
                       'established','establishment', 'our school began', 'we began',
                       'doors opened', 'school opened', 'about us', 'our school', 'who we are',
                       'identity', 'identities', 'profile', 'highlights']


charter_path = '../../charters_full_2015.pkl'
df_charter = pd.read_pickle(charter_path)
df_charter[['WEBTEXT','CMO_WEBTEXT']]=df_charter[['WEBTEXT','CMO_WEBTEXT']].fillna('') # turn nan to empty list/string for future convenience

# Optimized dict_count attempt for cases where entries in 'custom_dict' have long word lengths

# precalculations
dict_words = [entry.split() for entry in keywords] # list words for each dict entry
dict_lengths = [len(x) for x in dict_words]
first_words = [x[0] for x in dict_words] # first words of each entry in dict

def dict_count1(text):
    words_list = re.split('\W+', text) # list of words in text
    # find indices where word in first_words matches word in words_list
    mask = [[word == entry for word in words_list] for entry in first_words]
    indices = np.transpose(np.nonzero(mask))
    count = 0
    for ind in indices:
        if ind[1] <= (len(words_list) - dict_lengths[ind[0]]) and dict_words[ind[0]] == words_list[ind[1] : ind[1] + dict_lengths[ind[0]]]:
            count+=1
    return count

# Repurposed Jaren Haber's dict_count and helper function in webparser_mp.py. Bug fixed on chunk building.
max_entry_length = max([len(entry.split()) for entry in keywords]) # Get length (in words) of longest entry in combined dictionary

def dict_count(text):

    """Performs dictionary analysis, returning number of dictionary hits found.
    Removes punctuation and stems the phrase being analyzed.
    Compatible with multiple-word dictionary elements."""

    counts = 0 # number of matches between text_list and custom_dict
    splitted_phrase = re.split('\W+', text) # Remove punctuation with regex that keeps only letters and spaces

    # Do dictionary analysis for word chunks of lengths max_entry_length down to 1
    for length in range(1, max_entry_length + 1):
        if len(splitted_phrase) < length:
            continue # If text chunk is shorter than length of dict entries being matched, there are no matches.
        for i in range(len(splitted_phrase) - length + 1):
            entry = ' '.join(splitted_phrase[i:i+length]) # Builds chunk of 'length' words without ending space
            if entry in keywords:
                counts += 1

    return counts

# hybrid approach

# separate keywords to be treated differently
small_keywords = []
large_keywords = []

for entry in keywords:
    small_keywords.append(entry) if len(entry.split()) < 3 else large_keywords.append(entry)

large_words = [entry.split() for entry in large_keywords] # list words for each large dict entry
large_lengths = [len(x) for x in large_words]
large_first_words = [x[0] for x in large_words] # first words of each large entry in dict

def dict_count2(text):

    """Hybrid of dict_count and dict_count1. Uses dict_count1 approach to count matches for entries with > 2 words in keywords.
    Uses dict_count approach for all other entries.
    """

    counts = 0 # number of matches between text_list and custom_dict
    splitted_phrase = re.split('\W+', text) # Remove punctuation with regex that keeps only letters and spaces

    # Do dictionary analysis for word chunks of lengths max_entry_length down to 1
    for length in range(1, 3):
        if len(splitted_phrase) < length:
            continue # If text chunk is shorter than length of dict entries being matched, there are no matches.
        for i in range(len(splitted_phrase) - length + 1):
            entry = ' '.join(splitted_phrase[i:i+length]) # Builds chunk of 'length' words without ending space
            if entry in keywords:
                counts += 1
    mask = [[word == entry for word in splitted_phrase] for entry in large_first_words]
    indices = np.transpose(np.nonzero(mask))
    for ind in indices:
        if ind[1] <= (len(splitted_phrase) - large_lengths[ind[0]]) and large_words[ind[0]] == splitted_phrase[ind[1] : ind[1] + large_lengths[ind[0]]]:
            counts+=1
    return counts

def filter_pages(school_pages):
    """Returns the list of page text with hit count at least min hit count. Note this eliminates the tuple representation in WEBTEXT

    Also filters out duplicate tuples in WEBTEXT using set. Does not account for case where tuple is not exactly the same but
    text itself is the same.
    school_pages: entry of 'webtext' column
    """
    pages = set([p[3] for p in school_pages])
    return [page for page in set(pages) if dict_count2(page)>=MIN_HITCOUNT]
    # return [page for page in set(school_pages) if dict_count2(page[3])>=MIN_HITCOUNT] # maintains tuples but does not handle case where tuple is different but text is same


print('Page filter start')
filtered_pages = []
start = time.time()
for i, row in enumerate(df_charter['WEBTEXT'].values):
    filtered_pages.append(filter_pages(row))
    if i%1000 == 0:
        end = time.time()
        print('Time Elapsed:{:f}, Percent Complete:{:f}'.format(end - start,i*100/len(df_charter)))
df_charter['FILTERED_TEXT'] = pd.Series(filtered_pages)
# df_charter['FILTERED_TEXT'] = df_charter['WEBTEXT'].apply(filter_pages) # create column containing filtered pages

ckpt_file_path = 'charters_full_2015_checkpoint1.pkl'
df_result.to_pickle(ckpt_file_path) # checkpoint file contains new column 'FILTERED_TEXT'
print('Completed text filtering. Saved checkpoint to charters_full_2015_checkpoint1.pkl')

df_charter['REPLACED'] = df_charter.astype(str)['FILTERED_TEXT'] == '[]' # need at least 1 page per school else take cmo page
df_right = df_charter.groupby('CMO_NAME')['REPLACED'].sum() > 0 # df to be merged to the right of df_charter
df_right.columns = ['CMO_REPLACED'] # CMO_REPLACED tells us whether the CMO contains a school that replaced its webtext
df_right.reset_index(level = ['CMO_NAME'])
df_right = df_right[['CMO_NAME', 'CMO_REPLACED']] # maybe not necessary
df_result = pd.merge(df_charter, df_right, how = 'left', on = ['CMO_NAME']) # now CMO_REPLACED tells us if the school belongs to a CMO
                                                               # that replaced one its schools webtexts
ckpt_file_path = 'charters_full_2015_checkpoint2.pkl'
df_result.to_pickle(ckpt_file_path)
print('Completed CMO_REPLACED')
