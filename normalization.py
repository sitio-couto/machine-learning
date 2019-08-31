import numpy as np

def normalize_data(data, choice=1, features=[1,2,3,4]):
    ''' Returns the normalized dataset.
    
        Parameters:
            data (array list): csv table with the dataset (without the header).
            choice (int): integer indicating the transformation to be used.
            features (int list): Indexes of the features to be normalized.

        Returns:
            data (array list):  transformed data (original data is lost).
    '''

    # Data to be colected
    ranges = []
    means = []
    maxs = []
    mins = []
    stds = []

    # Gathering necessary data
    features_list = [[d[f] for d in data] for f in features]
    for fl in features_list:
        means.append(np.mean(fl))
        maxs.append(max(fl))
        mins.append(min(fl))
        stds.append(np.std(fl))
        ranges.append(maxs[-1]-mins[-1])

    #### Transforming the dataset ####

    if choice == 1:
        # Min-max normalization
        for entry in data:
            for (i,f) in enumerate(features):
                entry[f] = (entry[f] - mins[i])/ranges[i]  
    elif choice == 2:
        # Standardization
        for entry in data:
            for (i,f) in enumerate(features):
                entry[f] = (entry[f] - means[i])/stds[i]
    elif choice == 3:
        # Mean normalization
        for entry in data:
            for (i,f) in enumerate(features):
                entry[f] = (entry[f] - means[i])/ranges[i] 

    print
    return data

# def rmse(X, T, Y):
#     return np.sqrt(sum((X.dot(X) - Y)**2)/2*Y.shape[0])
