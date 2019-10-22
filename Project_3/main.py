# MC886 - Machine Learning - UNICAMP
# Project 3 - Unsupervised Learning and Dimensionality Reduction
#
# Authors:
# Victor Ferreira Ferrari - RA 187890
# Vinicius Couto Espindola- RA 188115

import numpy as np
from pandas import read_csv
from keras.utils import to_categorical
import normalization as norm
import neural as nr

data = read_csv('Dataset/fashion-mnist_train.csv')
Y = data['label'].to_numpy()
X = data.drop('label', 1).to_numpy()

# Normalization
choice = 2
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice)
Y = to_categorical(Y)

print(X)
print(Y)

arc = [300, 100]
dense = nr.get_neural_network_model(arc)
dense.summary()
dense.fit(x=X, y=Y, epochs=30, batch_size=128, validation_split=0.1)
