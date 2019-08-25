from sys import argv, maxsize
from collections import Counter
import numpy as np
from numpy import sqrt
from numpy import array as arr
import matplotlib.pyplot as plt 
import re
import pandas as pd
from datetime import date as date_check 

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

# Plots a relation between the average daily traffic per hour
def avg_traffic_hour_daily(data):
    # OBS: there are duplicated time stamps to separate more than one
    # enviromental condition (p.e. if its foggy and cloudy there will be duplicates)
    curr_day = 0
    traffic_hour = []
    for x in data[1:] :
        hour, day, _, _ = date_split(x[6]).values()
        if (day != curr_day):
            curr_day = day
            traffic_hour.append(arr([0]*24))
        traffic_hour[-1][hour] = x[-1]

    avg = sum(traffic_hour)/len(traffic_hour)
    plt.plot(range(0,24), avg)
    plt.title("Análise do tráfego por hora")
    plt.xlabel("Horas do dia (00h-24h)")
    plt.ylabel("Média diária de tráfego")
    plt.show()

    return

# Plot a realtion between weather descriptions and the average daily traffic per hour
def avg_traffic_per_weather(data):

    desc_main = {y:[0,0] for y in list(set([x[5] for x in data[1:]]))}
    desc = {y:[0,0] for y in list(set([x[6] for x in data[1:]]))}
    
    for x in data[1:]:
        desc_main[x[5]][0] += 1
        desc_main[x[5]][1] += int(x[-1])
        desc[x[6]][0] += 1
        desc[x[6]][1] += int(x[-1])

    counts = [x[0] if x[0]>0 else 1 for x in desc_main.values()]
    sums = [x[1] for x in desc_main.values()]
    desc_main_avg = arr(sums)/arr(counts)

    counts = [x[0] if x[0]>0 else 1  for x in desc.values()]
    sums = [x[1] for x in desc.values()]
    desc_avg = arr(sums)/arr(counts)
    
    a = plt.figure(1)
    plt.plot(list(desc.keys()), desc_avg)
    plt.xticks(list(desc.keys()), rotation='vertical')
    plt.margins(0)
    plt.subplots_adjust(bottom=0.5)
    plt.title("Análise do tráfego por descrição específica do clima")
    plt.xlabel("Descrições específicas do clima")
    plt.ylabel("Média diária de tráfego")
    
    b = plt.figure(2)
    plt.plot(list(desc_main.keys()), desc_main_avg)
    plt.title("Análise do tráfego por descrição geral do clima")
    plt.xlabel("Descrições gerais do clima")
    plt.ylabel("Média diária de tráfego")
  
    plt.show()
    input()
    
    return


def process_input(data):
    ''' Remove and re-format input data.

        Parameters:
            data (array list): csv table with the dataset header and content.
            
        Returns:
            data_frame (array list):  processed input data.
    '''

    # Alter and remove data
    for (i,d) in enumerate(data[1:], start=1):
        # Cast weather description to lower case
        data[i][6] = d[6].lower()

        # Propagate holidays and cast them to binary
        if d[0] == 'None' : 
            # In case theres no holiday
            data[i][0] = 0.0 
        elif d[0] != 1.0:
            # In case theres a non-binary holyday
            # Count how many holidays flags are missing
            j = i
            day = curr_day = date_split(d[-2])['d']
            while j < len(data) :
                curr_day = date_split(data[j][-2])['d']
                if day != curr_day : break
                else : j += 1

            # Iterate through timestamps binning holidays
            for k in range(i,j) : data[k][0] = 1.0

    # Discrete variables lists
    weekdays = {0:"mon", 1:"tue", 2:"wed", 3:"thu", 4:"fry", 5:"sat", 6:"sun"}
    desc_list = list(set([x[6] for x in data[1:]])) # Get unique values
    hour_list = list(map(str,range(0,24)))
    
    # Defining the new header and frame for processed data
    new_head = data[0][0:5] + desc_list + hour_list + list(weekdays.values()) + [data[0][-1][0:-1]]
    data_frame = [new_head]

    # Indexes to be casted
    cast_float = [1,2,3,4,8]

    for i in range(1,len(data)):
        # Ignore data with 0 kelvin
        if float(data[i][1]) == 0 : continue
        # Ignore duplicate hours (due to multiple weather description)
        if data[i-1][-2] == data[i][-2] : continue
        # Ignore ridiculous ammounts of rain
        if float(data[i][2]) > 300 : continue

        # Add row to dataframe
        data_frame.append([0.0]*len(new_head)) 

        # Cast numeric values read as string
        for j in cast_float : data[i][j] = float(data[i][j]) 
      
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

def normalize_data(data, choice=1):
    ''' Returns the normalizad dataset.
    
        Parameters:
            data (array list): csv table with  the dataset header and content.
            choice (int): integer indicating the transformation to be used.

        Returns:
            data (array list):  transformed data (original data is lost).
    '''
    
    # Indexes of features to be normalized
    features = [1,2,3,4]

    # Data to be colected
    ranges = []
    means = []
    maxs = []
    mins = []
    stds = []

    # Gathering necessary data
    features_list = [[d[f] for d in data[1:]] for f in features]
    for fl in  features_list:
        means.append(np.mean(fl))
        maxs.append(max(fl))
        mins.append(min(fl))
        stds.append(np.std(fl))
        ranges.append(maxs[-1]-mins[-1])

    #### Transforming the dataset ####

    if choice == 1:
        # Min-max normalization
        for entry in data[1:]:
            for (i,f) in enumerate(features):
                entry[f] = (entry[f] - mins[i])/ranges[i]  
    elif choice == 2:
        # Standardization
        for entry in data[1:]:
            for (i,f) in enumerate(features):
                entry[f] = (entry[f] - means[i])/std_dv[i]
    elif choice == 3:
        # Mean normalization
        for entry in data[1:]:
            for (i,f) in enumerate(features):
                entry[f] = (entry[f] - means[i])/ranges[i] 

    return data


def prepare_dataset():
    ''' Reads and prepares a dataset for regression.
    
        Returns:
            X (array list): coeficients matrix with label.
            Y (array list): results matrix with label.
    '''

    datafile = open(argv[1])
    data = list(map(lambda x : x.split(","), datafile.readlines()))

    data = process_input(data)
    data = normalize_data(data)
    
    # Separate processed data in coeficients and results
    X = data
    Y = [[x.pop()] for x in data]

    return X, Y

####################

X, Y = prepare_dataset()

# # Check
# for (i,t) in enumerate(Y):
#     if t[0] == 5969:
#         print("---(",i,")----------------------------")
#         list(map(print,zip(X[0],X[i])))
#         print(list(zip(Y[0],Y[i]))[0])

####################
