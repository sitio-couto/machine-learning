import numpy as np

def init_coefs(m, rand_seed=None):
    rand = np.random.RandomState(seed=rand_seed)
    return np.sqrt(2/(m + 1)) * rand.randn(m, 1)
