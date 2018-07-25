import pandas as pd
import time
import re
import numpy as np


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
df_charter['WEBTEXT']=df_charter['WEBTEXT'].fillna('') # turn nan to empty list/string for future convenience
df_charter['CMO_WEBTEXT'] = df_charter['CMO_WEBTEXT'].fillna('0') # ugly hack so that we can apply literal_eval on column later
df_charter['CMO_WEBTEXT'] = df_charter['CMO_WEBTEXT'].apply(ast.literal_eval) # apply to whole column
df_charter['CMO_WEBTEXT'] = df_charter['CMO_WEBTEXT'].replace(0, '') # now all nan are '' in both WEBTEXT columns

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

def filter_pages(school_pages, MIN_HITCOUNT = 1):
    """Returns the list of page text with hit count at least min hit count.

    Also filters out duplicate text.
    school_pages: entry of 'webtext' column
    """
    pages = set([Page(p) for p in school_pages])
    return [(p.url, p.boo, p.depth, p.text) for p in pages if dict_count2(p.text)>=MIN_HITCOUNT]
    # return [page for page in set(school_pages) if dict_count2(page[3])>=MIN_HITCOUNT] # maintains tuples but does not handle case where tuple is different but text is same
def run_filter(type, MIN_HITCOUNT = 1):
    if type == 'w':
        print('WEBTEXT Page filter start. Min hitcount: {:d}'.format(MIN_HITCOUNT))
        filtered_pages = []
        start = time.time()
        for i, row in enumerate(df_charter['WEBTEXT'].values):
            filtered_pages.append(filter_pages(row, 2))
            if i%1000 == 0:
                end = time.time()
                print('Time Elapsed:{:f}, Percent Complete:{:f}'.format(end - start,i*100/len(df_charter)))
        df_charter['FILTERED_TEXT' + str(MIN_HITCOUNT)] = pd.Series(filtered_pages, index=df_charter.index)
    elif type == 'c':
        print('CMO_WEBTEXT Page filter start. Min hitcount: {:d}'.format(MIN_HITCOUNT))
        filtered_pages = []
        start = time.time()
        for i, row in enumerate(df_charter['CMO_WEBTEXT'].values):
            filtered_pages.append(filter_pages(row, 2))
            if i%1000 == 0:
                end = time.time()
                print('Time Elapsed:{:f}, Percent Complete:{:f}'.format(end - start,i*100/len(df_charter)))
        df_charter['CMO_FILTERED_TEXT' + str(MIN_HITCOUNT)] = pd.Series(filtered_pages, index=df_charter.index)

        ckpt_file_path = 'charters_full_2015{:s}{:d}_checkpoint1.pkl'.format(type,MIN_HITCOUNT)
        df_charter.to_pickle(ckpt_file_path) # checkpoint file contains new column 'FILTERED_TEXT'
        print('Completed text filtering 2. Saved checkpoint to ' + 'charters_full_2015{:s}{:d}_checkpoint1.pkl'.format(type,MIN_HITCOUNT))

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

run_filter('w', 2)
run_filter('c', 2)

df_charter['REPLACED'] = df_charter.astype(str)['FILTERED_TEXT2'] == '[]' # need at least 1 page per school else take cmo page
df_charter.loc[df_charter['REPLACED'].values,'FILTERED_TEXT2'] = df_charter.loc[df_charter['REPLACED'].values,'CMO_FILTERED_TEXT2']  # replace empty FILTERED_TEXT with corresponding CMO_FILTERED_TEXT
df_right = df_charter.groupby('CMO_NAME')['REPLACED'].sum() > 0 # df to be merged to the right of df_charter
df_right.columns = ['CMO_REPLACED'] # CMO_REPLACED tells us whether the CMO contains a school that replaced its webtext
df_right.reset_index(level = ['CMO_NAME'])
df_result = pd.merge(df_charter, df_right, how = 'left', on = ['CMO_NAME']) # now CMO_REPLACED tells us if the school belongs to a CMO
                                                               # that replaced one its schools webtexts
ckpt_file_path = 'charters_full_2015_checkpoint2.pkl'
df_result.to_pickle(ckpt_file_path)
print('Completed CMO_REPLACED')
