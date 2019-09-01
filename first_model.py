from datetime import date as date_check 
from holidays import UnitedStates
import numpy as np
import re

def date_split(string):
    ''' Read date-time in "yyyy-mm-dd hh:mm:ss" and cast to int.

        Parameters:
            string (string): string containing the mentioned format.

        Returns:
            (dictionary): contains date-time info indexed by initials.
    '''

    date = re.split("-|:| ", string)
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    hour = int(date[3])

    return {"h":hour, "d":day, "m":month, "y":year}

def process_input(data):
    ''' Remove and re-format input data.

        Parameters:
            data (array list): csv table with the dataset header and content.
            
        Returns:
            data_frame (array list):  processed input data.
    '''

    # Features list
    new_head = ['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'thunderstorm with light rain', 'shower snow', 'thunderstorm with heavy rain', 'drizzle', 'light intensity shower rain', 'proximity shower rain', 'thunderstorm', 'heavy snow', 'proximity thunderstorm with rain', 'thunderstorm with rain', 'proximity thunderstorm', 'squalls', 'few clouds', 'light rain and snow', 'smoke', 'scattered clouds', 'thunderstorm with light drizzle', 'sky is clear', 'very heavy rain', 'light intensity drizzle', 'broken clouds', 'snow', 'heavy intensity rain', 'sleet', 'thunderstorm with drizzle', 'heavy intensity drizzle', 'light snow', 'light shower snow', 'moderate rain', 'haze', 'shower drizzle', 'proximity thunderstorm with drizzle', 'fog', 'mist', 'overcast clouds', 'freezing rain', 'light rain', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', 'mon', 'tue', 'wed', 'thu', 'fry', 'sat', 'sun', 'traffic_volume']
    data_frame = [new_head]
    holidays = UnitedStates(state='MN')

    # Alter and remove data
    for (i,d) in enumerate(data[1:], start=1):
        # Cast weather description to lower case
        data[i][6] = d[6].lower()

        # Propagate holidays and cast them to binary
        if d[0] == 'None' : 
            # In case theres no holiday
            data[i][0] = 0.0 
        elif d[-2].split()[0] in holidays:
            # In case its a known minnesota holyday date
            data[i][0] = 1.0 
        else:
            # Unknown holidays are discarded for simplicity
            data[i][0] = 0.0

    # # Discrete variables lists
    weekdays = {0:"mon", 1:"tue", 2:"wed", 3:"thu", 4:"fry", 5:"sat", 6:"sun"}
    # desc_list = list(set([x[6] for x in data[1:]])) # Get unique values
    # hour_list = list(map(str,range(0,24)))
    
    # # Defining the new header and frame for processed data
    # new_head = data[0][0:5] + desc_list + hour_list + list(weekdays.values()) + [data[0][-1][0:-1]]

    # Indexes to be casted
    cast_float = [1,2,3,4,8]

    for i in range(1,len(data)):
        # Ignore data with 0 kelvin
        if float(data[i][1]) == 0 : continue
        # Ignore duplicate hours (due to multiple weather description)
        if data[i-1][-2] == data[i][-2] : continue
        # Ignore ridiculous ammounts of rain
        if float(data[i][2]) > 300 : continue
        # Cast numeric values read as string to float
        for j in cast_float : 
            data[i][j] = float(data[i][j]) 

        # Add row to dataframe
        data_frame.append([0.0]*len(new_head)) 
      
        # split and cast date
        hour, day, month, year = date_split(data[i][-2]).values()

        # Set discrete weekday variable
        day_name = weekdays[date_check(year,month,day).weekday()]
        data_frame[-1][new_head.index(day_name)] = 1.0

        # Set discrete hour variable
        data_frame[-1][new_head.index(str(hour))] = 1.0

        # Set discrete weather description variables
        j = i
        while j < len(data) and data[j][-2] == data[i][-2] :
            data_frame[-1][new_head.index(data[j][6])] = 1.0 
            j += 1

        # Set quantitative data
        data_frame[-1][-1] = data[i][-1]
        data_frame[-1][:5] = data[i][:5]     

    return data_frame

def prepare_dataset(set_name):
    ''' Reads and prepares a dataset for regression.
    
        Returns:
            X (array list): coeficients matrix with label.
            Y (array list): results matrix with label.
    '''

    datafile = open(set_name)
    data = list(map(lambda x : x.split(","), datafile.readlines()))

    data = process_input(data)
    
    # Separate processed data in coeficients and results
    X = data[1:] # Remove header
    Y = [[x.pop()] for x in X]

    # Returns processed input and indexes to be normalized
    return X, Y, [1,2,3,4]



# ############# TESTING AREA ###############
# prepare_dataset("Datasets/dataset_with_column_names.csv")
