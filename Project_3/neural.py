import numpy as np
from const import *
from keras import Model
from keras.layers import Dense, Input, Add, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

def get_neural_network_model(arc, activ='relu', optimizer='adam'):
    
    # Input
    inp = Input(shape=(IMG_HEIGHT*IMG_WIDTH, ))
    
    # Hidden Layers
    layer = inp
    for n in arc:
        layer = Dense(n, activation=activ, kernel_regularizer=l2(0.001))(layer)
    
    # Output
    out = Dense(N_CLASSES, activation='softmax', kernel_regularizer=l2(0.001))(layer)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
    
