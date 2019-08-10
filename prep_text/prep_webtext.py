def preprocess_wem(tuplist): # inputs were formerly:
    
    '''This function cleans and tokenizes sentences, removing punctuation and numbers and making words into lower-case stems.
    Inputs: list of four-element tuples, the last element of which holds the long string of text we care about;
        an integer limit (bypassed when set to -1) indicating the DF row index on which to stop the function (for testing purposes),
        and similarly, an integer start (bypassed when set to -1) indicating the DF row index on which to start the function (for testing purposes).
    This function loops over five nested levels, which from high to low are: row, tuple, chunk, sentence, word.
    Note: This approach maintains accurate semantic distances by keeping stopwords.'''
        
    global mpdo # Check if we're doing multiprocessing. If so, then mpdo=True
    global words_by_sentence # Grants access to variable holding a list of lists of words, where each list of words represents a sentence in its original order (only relevant for this function if we're not using multiprocessing)
    global pcount # Grants access to preprocessing counter
    
    known_pages = set() # Initialize list of known pages for a school

    if type(tuplist)==float:
        return # Can't iterate over floats, so exit
    
    #print('Parsing school #' + str(pcount)) # Print number of school being parsed

    for tup in tuplist: # Iterate over tuples in tuplist (list of tuples)
        if tup[3] in known_pages or tup=='': # Could use hashing to speed up comparison: hashlib.sha224(tup[3].encode()).hexdigest()
            continue # Skip this page if exactly the same as a previous page on this school's website

        for chunk in tup[3].split('\n'): 
            for sent in sent_tokenize(chunk): # Tokenize chunk by sentences (in case >1 sentence in chunk)
                sent = clean_sentence(sent, remove_stopwords=True) # Clean and tokenize sentence
                
                if ((sent == []) or (len(sent) == 0)): # If sentence is empty, continue to next sentence without appending
                    continue
                
                # Save preprocessing sentence to file (if multiprocessing) or to object (if not multiprocessing)
                if mpdo:
                    try: 
                        if (os.path.exists(wordsent_path)) and (os.path.getsize(wordsent_path) > 0): 
                            append_sentence(sent, wordsent_path) # If file not empty, add to end of file
                        else:
                            write_sentence(sent, wordsent_path) # If file doesn't exist or is empty, start file
                    except FileNotFoundError or OSError: # Handle common errors when calling os.path functions on non-existent files
                        write_sentence(sent, wordsent_path) # Start file
                
                else:
                    words_by_sentence.append(sent) # If not multiprocessing, just add sent to object
                    
                    
        known_pages.add(tup[3])
    
    pcount += 1 # Add to counter
    
    return