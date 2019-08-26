import csv
import os
import numpy as np
from math import floor

TEST_SIZE  = 0.2
VALID_SIZE = 0.1
FOLDER_NAME= 'Datasets'
VALID_FILE = 'validate.csv'
TEST_FILE  = 'test.csv'
TRAIN_FILE = 'training.csv'

def split_sets(filename):
	'''
		Splits dataset between TRAINING, VALIDATION and TEST sets.
	'''
	
	# Reading csv into numpy array
	f = open(filename, 'r')
	data = list(csv.reader(f, delimiter=','))
	f.close()
	
	data = np.array(data)
	
	# Amount of examples (+ first row)
	amount = np.size(data, 0)
	
	# Getting validation set from random positions and saving to CSV file
	valid_idx = np.random.randint(low=0, high=(amount-1), size=(floor(amount*VALID_SIZE)))
	valid = data[valid_idx]
	np.savetxt(os.path.join(FOLDER_NAME, VALID_FILE), valid, delimiter=',', fmt='%s')
	
	# Removing validation data from data and getting new number of rows
	data = np.delete(data, valid_idx, 0)
	new_amount = np.size(data, 0)
	
	# Getting test set from random positions and saving to CSV file
	test_idx = np.random.randint(low=0, high=(new_amount-1), size=(floor(amount*TEST_SIZE)))
	test = data[test_idx]
	np.savetxt(os.path.join(FOLDER_NAME, TEST_FILE), test, delimiter=',', fmt='%s')
	
	# Removing test data and saving training set to CSV file
	data = np.delete(data, test_idx, 0)
	np.savetxt(os.path.join(FOLDER_NAME, TRAIN_FILE), data, delimiter=',', fmt='%s')
	
	# WARNING: FIRST LINE STILL IN ONE OF THE FILES, REMOVE COLUMN NAMES FIRST
		
