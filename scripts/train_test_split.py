#!/usr/bin/env python
# coding: utf-8

# Train/test split

# Author: Jaren Haber, PhD Candidate
# Project: Charter school identities
# Institution: University of California, Berkeley, Dept. of Sociology
# Date: August 27, 2018
# This script splits the full charter data set (including school and community variables, including WEBTEXT) into a training set and test set using an 80/20 split (per Pareto). 


## Import key packages

import pandas as pd # for working with dataframes
import gc # For managing garbage collector (to increase efficiency loading large files into memory)
from sklearn.model_selection import train_test_split # For splitting into train/test set
import sys # For terminal tricks

## Define file paths
charters_path = "../../nowdata/charters_full_2015_250_new_counts.pkl" # All text data; only charter schools (regardless if open or not)
train_path = "../../nowdata/traincf_2015_250_new_counts.pkl" # Training set (80% of data)
test_path = "../../nowdata/ignore/testcf_2015_250_new_counts.pkl"

# Load charter data into DF
gc.disable() # disable garbage collector
df = pd.read_pickle(charters_path)
gc.enable() # enable garbage collector again

# Split data into 80% for training and 20% for test using random sample
proptest = 0.2 # fraction of 0.2 = 20% test set
print("Creating " + str((1-proptest)*100) + "/" + str(proptest*100) + " train/test split...")
traindf, testdf = train_test_split(df, test_size = proptest, random_state = 0)

# Save data for later use
traindf.to_pickle(train_path)
testdf.to_pickle(test_path)

print("Training set saved to " + str(train_path))
print("Test set saved to " + str(test_path))

sys.exit() # Kill script when done, just to be safe
