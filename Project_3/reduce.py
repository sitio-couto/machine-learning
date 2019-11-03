from sklearn.decomposition import PCA
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import Model

def reduce_PCA(X, variance):
    pca = PCA(variance)
    pca.fit(X)
    
    return pca, pca.transform(X)

# Dense autoencoder
def autoencoder(X, arc, inp_shape, activ='relu'):
    
    # Input
    inp = Input(shape=inp_shape)
    
    # Hidden Layers - Encoder
    encoder = inp
    for n in arc:
        encoder = Dense(n, activation=activ)(encoder)
    
    # Hidden Layers - Decoder
    decoder = encoder
    for i in range(len(arc)-1)[::-1]:
        decoder = Dense(arc[i], activation=activ)(decoder)
    
    # Output
    out = Dense(inp_shape[0], activation='sigmoid')(decoder)
    
    # Get models
    autoencoder = Model(inputs=inp, outputs=out)
    encoder = Model(inputs=inp, outputs=encoder)
    autoencoder.compile(loss='mean_squared_error', optimizer='RMSProp')
    
    return autoencoder, encoder
    
# Convolutional autoencoder
def autoencoder_conv(X, arc, filt_size, pool_size, activ='relu'):
    
    # Input
    inp = Input(shape=(X.shape[1], X.shape[2],1))
    
    # Hidden Layers - Encoder
    encoder = inp
    for n in arc:
        encoder = Conv2D(n, filt_size, activation=activ, padding='same')(encoder)
        encoder = MaxPooling2D(pool_size, padding='same')(encoder)
    
    shape = encoder.get_shape().as_list()
    encoder = Flatten()(encoder)
    
    # Hidden Layers - Decoder
    decoder = Reshape((shape[1],shape[2],shape[3]))(encoder)
    for i in range(len(arc))[::-1]:
        decoder = Conv2D(arc[i], filt_size, activation=activ, padding='same')(decoder)
        decoder = UpSampling2D(pool_size)(decoder)
    
    # Output
    out = Conv2D(1, filt_size, activation='sigmoid', padding='same')(decoder)
    
    # Get models
    autoencoder = Model(inputs=inp, outputs=out)
    encoder = Model(inputs=inp, outputs=encoder)
    autoencoder.compile(loss='mean_squared_error', optimizer='adadelta')
    
    return autoencoder, encoder
