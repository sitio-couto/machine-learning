import neural as nr
import numpy as np

def run_network(X, Y, arc, inp_shape, n_out, epochs=50, batch_size=128, val_split=0.1):
    dense = nr.get_neural_network_model(arc, inp_shape, n_out)

    # Train network
    dense.summary()
    dense, hist = nr.train(dense, X, Y, epochs=epochs, batch_size=batch_size, val_split=val_split)
    print('Validation Accuracy: ', np.max(hist['val_accuracy']))
