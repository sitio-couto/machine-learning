import numpy as np

def get_stats(data, choice=1):
    means,stds,mins,ranges = [],[],[],[]
    
    # Stats for normalization
    if choice == 1 or choice == 3:
        mins = np.apply_along_axis(np.amin, 0, data).astype('int16')
        maxs = np.apply_along_axis(np.amax, 0, data).astype('int16')
        ranges = maxs - mins
    if choice == 2 or choice == 3:
        means = np.apply_along_axis(np.mean, 0, data).astype('float16')
    if choice == 2:
        stds = np.apply_along_axis(np.std, 0, data).astype('float16')
    
    return {'mean':means, 'std':stds, 'mins':mins, 'range':ranges}
    
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
        data = np.apply_along_axis(lambda x: (x - stats['mins'])/stats['range'], 1, data)
            
    # Standardization
    elif choice == 2:
        data = np.apply_along_axis(lambda x: (x - stats['mean'])/stats['std'], 1, data)
            
    # Mean normalization
    elif choice == 3:
        data = np.apply_along_axis(lambda x: (x - stats['mean'])/stats['range'], 1, data)

    return data