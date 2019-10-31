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

# Deep dense neural net
# arc = [470,284,169,100,60,30]
arc = [300]
start = time.process_time()
run.run_network(X, Y, arc, (IMG_HEIGHT*IMG_WIDTH, ), N_CLASSES , epochs=30, batch_size=1024, val_split=0.1)
print("Full Runtime:", time.process_time() - start)

encoders = [[255], [136], [136,81], [136,50], [128,64,32], [255,136,20]]
for arc_enc in encoders:
    # Dense Auto-Encoder
    start = time.process_time()
    autoenc, enc = red.autoencoder(X, arc_enc, (IMG_HEIGHT*IMG_WIDTH, ))
    autoenc, _ = nr.train(autoenc, X, X, epochs=50, batch_size=1024, val_split=0, use_calls=False, best=False)
    print("Encoder Runtime:", time.process_time() - start)
    Xenc = enc.predict(X)
    print(f"{X.shape[1]} => {Xenc.shape[1]}")
    # Training
    run.run_network(Xenc, Y, arc, (arc_enc[-1], ), N_CLASSES , epochs=30, batch_size=1024, val_split=0.1)
    print("Full Runtime:", time.process_time() - start)
    print("============================================================================================")
