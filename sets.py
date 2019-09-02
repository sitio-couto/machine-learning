import csv
import os
import numpy as np
from random import sample
from math import floor

FOLDER_NAME= 'Datasets'
VALID_FILE = 'validate.csv'
TEST_FILE  = 'test.csv'
TRAIN_FILE = 'training.csv'

def split_sets(filename, test_sz=0.1, valid_sz=0.1):
    '''
        Splits dataset between TRAINING, VALIDATION and TEST sets.
    '''
    
    # Reading csv into numpy array
    f = open(filename, 'r')
    data = list(csv.reader(f, delimiter=','))
    f.close()
    
    # Header
    data = np.array(data)
    header = data[0]
    data = data[1:]
    
    # Amount of examples (+ first row)
    amount = np.size(data, 0)
    
    # Getting validation set from random positions and saving to CSV file
    valid_idx = np.array(sample(range(amount), floor(amount*valid_sz)))
    valid = np.vstack((header, data[valid_idx]))
    np.savetxt(os.path.join(FOLDER_NAME, VALID_FILE), valid, delimiter=',', fmt='%s')
    
    # Removing validation data from data and getting new number of rows
    data = np.delete(data, valid_idx, 0)
    new_amount = np.size(data, 0)
    
    # Getting test set from random positions and saving to CSV file
    test_idx = np.array(sample(range(new_amount), floor(amount*test_sz)))
    test = np.vstack((header, data[test_idx]))
    np.savetxt(os.path.join(FOLDER_NAME, TEST_FILE), test, delimiter=',', fmt='%s')
    
    # Removing test data and saving training set to CSV file
    data = np.delete(data, test_idx, 0)
    data = np.vstack((header, data))
    np.savetxt(os.path.join(FOLDER_NAME, TRAIN_FILE), data, delimiter=',', fmt='%s')

