from sklearn.decomposition import PCA
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
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
    autoencoder.compile(loss='mean_squared_error', optimizer='RMSProp', metrics=['accuracy'])
    
    return autoencoder, encoder
    
# Convolutional autoencoder
def autoencoder_conv(X, arc, filt_size, pool_size, activ='relu'):
    
    # Input
    inp = Input(shape=X.shape)
    
    # Hidden Layers - Encoder
    encoder = inp
    for n in arc:
        encoder = Conv2D(n, filt_size, activation=activ, padding='same')(encoder)
        encoder = MaxPooling2D(pool_size, padding='same')(encoder)
    
    # Hidden Layers - Decoder
    decoder = encoder
    for i in range(len(arc)-1)[::-1]:
        decoder = Conv2D(n, filt_size, activation=activ, padding='same')(decoder)
        decoder = UpSampling2D(pool_size, padding='same')(decoder)
    
    # Output
    out = Conv2D(1, filt_size, activation='sigmoid', padding='same')(decoder)
    
    # Get models
    autoencoder = Model(inputs=inp, outputs=out)
    encoder = Model(inputs=inp, outputs=encoder)
    autoencoder.compile(loss='mean_squared_error', optimizer='RMSProp', metrics=['accuracy'])
    
    return autoencoder, encoder
