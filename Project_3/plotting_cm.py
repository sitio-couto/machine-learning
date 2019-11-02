import numpy as np
import visualization as vis
import misc
from const import *

path = "Confusion Matricies/aggclust/"
train = "cm_train_aggclus_"
valid = "cm_val_aggclus_"
pcas = ["0.95pca", "0.9pca", "0.85pca","0.8pca", "0.75pca", "0.7pca"]

train_cms = []
valid_cms = []
for x in pcas:
    train_cms.append(np.load(path+train+x+'.npy'))
    valid_cms.append(np.load(path+valid+x+'.npy'))

classes = list(CLASS_NAMES.values())

for i,p in enumerate(pcas):
    print("Model: "+p)
    print(f"Training Accuracy: {misc.accuracy(train_cms[i])}")
    print(f"Validation Accuracy: {misc.accuracy(valid_cms[i])}") 
    vis.plot_confusion_matrix(train_cms[i], classes, f"Agglomerative Clustering ({p})")