import numpy as np
import pandas as pd
    
def prepare_dataset(set_path, type='train'):
    
    # Reading file to dataframe
    df = pd.read_csv(set_path)
    
    # Cleaning
    df = process_input(df)
    
    Y = df['traffic_volume'].T
    X = df.drop(axis=1, labels='traffic_volume')
    
    return X.values,Y.values,[1,2,3,4]

def process_input(data):
    ''' Remove and re-format input data.
        Receives pandas dataframe with all data.
        Returns processed data.
    '''
    # Removes 0 kelvin values, ridiculous amounts of rain and duplicate measurements for same time.
    data.drop(data[(data['temp'] == 0) | (data['rain_1h'] > 300)].index, inplace = True)
    data.drop_duplicates('date_time', inplace=True)

    # Get holidays
    data['holiday'] = np.where(data['holiday'] == 'None', 0, 1)
    
    # Get day, hour, month, year
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].dt.hour
    data['day'] = data['date_time'].dt.day
    data['month'] = data['date_time'].dt.month
    data['year'] = data['date_time'].dt.year
    
    # Get weekend
    data['weekday'] = data['date_time'].dt.weekday
    data['weekday'] = np.where((data['weekday']) == 5 | (data['weekday'] == 6), 1, 0)
    
    data.drop(labels='date_time', axis=1, inplace=True)
    
    # Get peak and commercial times
    peak = list(range(15,18))
    comm = list(range(6, 19))
    data['peak'] = np.where((data['hour'] >= peak[0]) & (data['hour'] <= peak[-1]), 1, 0)
    data['commercial'] = np.where((data['hour'] >= comm[0]) & (data['hour'] <= comm[-1]), 1, 0)
    
    # Weather variables
    weather_main = ['Clear', 'Clouds', 'Rain', 'Drizzle', 'Fog', 'Mist', 'Haze', 'Thunderstorm', 'Snow', 'Smoke', 'Squall']
    
    for i in weather_main:
        data[i] = np.where(data['weather_main'] == i, 1, 0)
            
    data.drop(labels=['weather_description', 'weather_main'], axis=1, inplace=True)
    print(data)
    return data
    
