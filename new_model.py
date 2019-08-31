import numpy as np
import pandas as pd

def prepare_dataset(set_path, type='train'):
	
	# Reading file to dataframe
	df = pd.read_csv(set_path)
	
	# Cleaning
	#df = process_input(df)
	
	Y = df['traffic_volume']
	X = df.drop(axis=1, labels='traffic_volume')
	return X.values,Y.values
	
