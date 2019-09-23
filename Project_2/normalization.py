import numpy as np

def normalize_data(data, choice=1):
    ''' Returns the normalized dataset.
    
        Parameters:
            data (array list): numpy array with the dataset.
            choice (int): integer indicating the transformation to be used.

        Returns:
            data (array list):  transformed data (original data is lost).
    '''

    #### Transforming the dataset ####
    # Min-max normalization
    if choice == 1:
        mins = np.apply_along_axis(np.amin, 0, data)
        maxs = np.apply_along_axis(np.amax, 0, data)
        ranges = maxs - mins
        data = np.apply_along_axis(lambda x: (x - mins)/ranges, 1, data)
            
    # Standardization
    elif choice == 2:
        means = np.apply_along_axis(np.mean, 0, data)
        stds = np.apply_along_axis(np.std, 0, data)
        data = np.apply_along_axis(lambda x: (x - means)/stds, 1, data)
            
    # Mean normalization
    elif choice == 3:
        means = np.apply_along_axis(np.mean, 0, data)
        mins = np.apply_along_axis(np.amin, 0, data)
        maxs = np.apply_along_axis(np.amax, 0, data)
        ranges = maxs - mins
        data = np.apply_along_axis(lambda x: (x - means)/ranges, 1, data)

    return data
