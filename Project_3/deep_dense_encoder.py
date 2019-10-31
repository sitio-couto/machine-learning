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

data = read_csv('Dataset/fashion-mnist_train.csv')
Y = data['label'].to_numpy()
X = data.drop('label', 1).to_numpy()
classes = list(CLASS_NAMES.values())

# Normalization
choice = 2
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice).astype('float32')
Y = to_categorical(Y).astype('int8')

arcs = [[255], [136], [136,81], [136,50], [255,136,32], [255,136,20]]
for arc_enc in arcs:
    # Dense Auto-Encoder
    autoenc, enc = red.autoencoder(X, arc_enc, (IMG_HEIGHT*IMG_WIDTH, ))
    autoenc, _ = nr.train(autoenc, X, X, epochs=50, batch_size=1024, val_split=0, use_calls=False, best=False)
    Xenc = enc.predict(X)
    print(f"{X.shape[1]} => {Xenc.shape[1]}")
    # Training
    run.run_network(Xenc, Y, arc, (arc_enc[-1], ), N_CLASSES , epochs=30, batch_size=1024, val_split=0.1)
    print("============================================================================================")