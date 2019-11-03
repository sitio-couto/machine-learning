import numpy as np
import visualization as vis
import normalization as norm
import misc
from const import *
from pandas import read_csv

data = read_csv('Dataset/fashion-mnist_train.csv')
Y = data['label'].to_numpy()
X = data.drop('label', 1).to_numpy()
classes = list(CLASS_NAMES.values())


# Normalization
choice = 2
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice).astype('float32')

vis.pca_plotting(X, Y, classes, title="PCA Visualization (True Labels)", s=2)