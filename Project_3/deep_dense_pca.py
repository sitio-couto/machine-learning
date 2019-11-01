import numpy as np
from pandas import read_csv
from keras.utils import to_categorical
from const import *
import normalization as norm
import neural as nr
import reduce as red
import clustering as clus
import visualization as vis
import misc
import run
import time


data = read_csv('Dataset/fashion-mnist_train.csv')
Y = data['label'].to_numpy()
X = data.drop('label', 1).to_numpy()
classes = list(CLASS_NAMES.values())

# Normalization
choice = 2
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice).astype('float32')
Y = to_categorical(Y).astype('int8')


# Deep dense neural net architecture
# arc = [470,284,169,100,60,30]
arc = [300]

# Deep dense net neural net
variance = [1, .95, .90, .85, .80, .75, .70]
for v in variance:
    start = time.process_time()
    if not v == 1 : 
        pca, Xpca = red.reduce_PCA(X, v)
        print("PCA Runtime:", time.process_time() - start)
        n_comp = pca.n_components_
        print('Components:', n_comp)
        print(f'Preserving {v}% of the data.')
    else : 
        Xpca = X
        n_comp = X.shape[1]
        print("No PCA applied")

    # Train with PCA
    run.run_network(Xpca, Y, arc, (n_comp, ), N_CLASSES , epochs=30, batch_size=1024, val_split=0.1)
    print("Full Runtime:", time.process_time() - start)
    print("=======================================================================================")