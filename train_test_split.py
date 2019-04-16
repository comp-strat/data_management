#!/usr/bin/env python
# coding: utf-8

# Train/test split

# Author: Jaren Haber, PhD Candidate
# Project: Charter school identities
# Institution: University of California, Berkeley, Dept. of Sociology
# Date created: August 27, 2018
# Date last modified: November 29, 2018
# This script splits the full charter data set (including school and community variables, with WEBTEXT) into a training set and test set using an 80/20 split (as per Pareto). 


## Import key packages

import pandas as pd # for working with dataframes
import gc # For managing garbage collector (to increase efficiency loading large files into memory)
from sklearn.model_selection import train_test_split # For splitting into train/test set
import sys # For terminal tricks

## Define file paths
# Inputs:
# Latest: "../../nowdata/backups/charters_full_2015_250_v2a_unlappedtext_counts3_geoleaid.pkl"
charters_path = "../../nowdata/charters_2015.pkl" # All text data; only charter schools (regardless if open or not)
stats_path = "../../nowdata/backups/charters_stats_2015_v2a.csv" # Quantitative data for statistics etc. (no text)

# Outputs:
train_path = "../../nowdata/stats_traincf_2015_v2a.csv" # Training set (80% of data) for model development
test_path = "../../nowdata/ignore/stats_testcf_2015_v2a.csv" # Test set (20%) for later use
share_path = "../../wem4themes/charter_data_201516.csv" # For sharing data @ workshops, GitHub, etc.

## Load charter data into DF
gc.disable() # disable garbage collector
df = pd.read_csv(stats_path, low_memory=False)
gc.enable() # enable garbage collector again

## Split data into 80% for training and 20% for test using random sample
proptest = 0.4 # fraction of 0.4 = 40% test set
print("Creating " + str((1-proptest)*100) + "/" + str(proptest*100) + " train/test split...")
traindf, testdf = train_test_split(df, test_size = proptest, random_state = 0)

## Save data for later use
traindf.to_csv(train_path, index=False)
traindf.to_csv(share_path, index=False)
testdf.to_csv(test_path, index=False)

print("Training set saved to " + str(train_path) + " and " + str(share_path))
print("Test set saved to " + str(test_path))

sys.exit() # Kill script when done, just to be safe
