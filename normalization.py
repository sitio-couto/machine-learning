from sys import argv, maxsize
from collections import Counter
import numpy as np
from numpy import array as arr
import matplotlib.pyplot as plt 
import re
import pandas as pd
from datetime import date as date_check 

def date_split(string):
    date = re.split("-|:| ", string)
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    hour = int(date[3])

    return hour, day, month, year

# Plots a relation between the average daily traffic per hour
def avg_traffic_hour_daily(data):
    # OBS: there are duplicated time stamps to separate more than one
    # enviromental condition (p.e. if its foggy and cloudy there will be duplicates)
    curr_day = 0
    traffic_hour = []
    for x in data[1:] :
        hour, day, _, _ = date_split(x[6])
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


def process_data(data):

    # Alter and remove data
    for i in range(1,len(data)):
        # Remove main weather description
        data[i][6] = data[i][6].lower()

        # Propagate and normalize holidays
        if data[i][0] == 'None' : data[i][0] = 0.0
        elif data[i][0] != 1.0:
            day = curr_day = int(re.split("-|:| ", data[i][-2])[2])
            
            j = i
            while j < len(data) :
                curr_day = int(re.split("-|:| ", data[j][-2])[2])
                if day != curr_day : break
                else : j += 1

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
        date = re.split("-|:| ", data[i][-2])
        year = int(date[0])
        month = int(date[1])
        day = int(date[2])
        hour = int(date[3])

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

        # Quantitative data
        data_frame[-1][-1] = data[i][-1]
        data_frame[-1][:5] = data[i][:5]     

    return data_frame

def normalize_data(data):
    # Indexes of features to be scaled
    features = [1,2,3,4]
    
    # Values for mean normalization
    means = [0]*len(features)
    ranges = [(maxsize, -maxsize)]*len(features)

    # Calculate means and reanges of the features to be scaled
    for entry in data[1:]:
        for (i,f) in enumerate(features):
            means[i] += entry[f]
            ranges[i] = (min(entry[f], ranges[i][0]),max(entry[f], ranges[i][1]))
    means = [x/len(data) for x in means]
    ranges = [x[1]-x[0] for x in ranges]
    print("Medias=>",means)
    print("Ranges=>",ranges)

    # Feature scaling
    for entry in data[1:]:
        for (i,f) in enumerate(features):
            entry[f] = entry[f]/ranges[i] 

    # # Mean normalization
    # for entry in data[1:]:
    #     for (i,f) in enumerate(features):
    #         entry[f] = (entry[f] - means[i])/ranges[i] 

    return data

def read_and_normalize():
    datafile = open(argv[1])
    data = list(map(lambda x : x.split(","), datafile.readlines()))

    data = process_data(data)
    data = normalize_data(data)
    
    for d in data:
        if d[-1] == 5969:
            print("--(",data.index(d),")--------------------")
            list(map(print, zip(data[0], d)))
            exit(1)

####################
read_and_normalize()
####################
