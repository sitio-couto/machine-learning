import numpy as np
from keras import Model
from keras.models import load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from os.path import exists

# Simple neural network, generic
def get_neural_network_model(arc, inp_shape, n_out, activ='relu', optimizer='adam'):
    
    # Input
    inp = Input(shape=inp_shape)
    
    # Hidden Layers
    layer = inp
    for n in arc:
        layer = Dense(n, activation=activ, kernel_regularizer=l2(0.001))(layer)
    
    # Output
    out = Dense(n_out, activation='softmax', kernel_regularizer=l2(0.001))(layer)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Train the neural network
def train(model, X, Y, epochs=50, batch_size=128, val_split=0.1, patience=5):
    callbacks = [
                 EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience),
                 ModelCheckpoint(monitor='val_loss', filepath='best_model.h5', save_best_only=True)
                ]
    
    history = model.fit(x=X, y=Y, epochs=epochs, batch_size=batch_size, validation_split=val_split, callbacks=callbacks).history
    
    if exists('best_model.h5'):
        model = load_model('best_model.h5')
    
    return model, history
    
