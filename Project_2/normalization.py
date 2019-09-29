import numpy as np

def get_stats(data, choice=1):
    means,stds,mins,ranges = [],[],[],[]
    
    # Stats for normalization
    if choice == 1 or choice == 3:
        mins = np.apply_along_axis(np.amin, 0, data)
        maxs = np.apply_along_axis(np.amax, 0, data)
        ranges = maxs - mins
    if choice == 2 or choice == 3:
        means = np.apply_along_axis(np.mean, 0, data)
    if choice == 2:
        stds = np.apply_along_axis(np.std, 0, data)
    
    return np.array([means,stds,mins,ranges])
    
def normalize_data(data, stats, choice=1):
    ''' Returns the normalized dataset.
    
        Parameters:
            data (array) : numpy array with the dataset.
            stats (array): numpy array with stats given by "get_stats". 0: means. 1: stds. 2:mins. 3:ranges
            choice (int) : integer indicating the transformation to be used.

        Returns:
            data (array list):  transformed data (original data is lost).
    '''

    #### Transforming the dataset ####
    # Min-max normalization
    if choice == 1:
        data = np.apply_along_axis(lambda x: (x - stats[2])/stats[3], 1, data)
            
    # Standardization
    elif choice == 2:
        data = np.apply_along_axis(lambda x: (x - stats[0])/stats[1], 1, data)
            
    # Mean normalization
    elif choice == 3:
        data = np.apply_along_axis(lambda x: (x - stats[0])/stats[3], 1, data)

    return data

def out_layers(array):
    '''Converts values from a list to one hot enconded output layers

        Parameters:
            array (list of ints): Contains the integer ID for each class
    
        Returns:
            np.ndarray : 2D array where each colum is a output layer
    '''
    lower = min(array)
    upper = max(array)
    lines = upper-lower+1
    new_arr = np.zeros((lines,len(array)))

    for j,i in enumerate(map(lambda x : x-lower, array)):
       new_arr[i,j] = 1

    return new_arr