from sys import argv
from collections import Counter
import numpy as np
from numpy import array as arr
import matplotlib.pyplot as plt 
import re
import pandas as pd
import datetime

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


def remove_and_cast_features(data):

    # Defining the new header for processed data
    desc_list = list(set([x[6] for x in data[1:]])) # Get unique values
    hour_list = list(map(str,range(0,24)))
    new_head = data[0][0:5] + desc_list + hour_list + [data[0][-1][0:-1]]
    data_frame = [new_head]

    for i in range(1,len(data)):
        if data[i][0] == 'None' : data[i][0] = 0.0
        elif data[i][0] != 1.0:
            day = curr_day = int(re.split("-|:| ", data[i][-2])[2])
            
            j = i
            while j < len(data) :
                curr_day = int(re.split("-|:| ", data[j][-2])[2])
                if day != curr_day : break
                else : j += 1

            for k in range(i,j) : data[k][0] = 1.0


    print(list(map(print,data)))

    cast_float = [1,2,3,4,8]

    for i in range(1,len(data)):
        # Add row to dataframe
        data_frame.append([0.0]*len(new_head)) 

        # Cast numeric values read as string
        for j in cast_float : data[i][j] = float(data[i][j]) 
      
        # split and cast date
        date = re.split("-|:| ", data[i][-2])
        year = date[0]
        month = date[1]
        day = date[2]
        hour = str(int(date[3]))

        # Set discrete hour varible
        data_frame[i][new_head.index(hour)] = 1.0

        # Set discrete classification data varible
        data_frame[i][new_head.index(data[i][6])] = 1.0

        # Quantitative data
        data_frame[i][-1] = data[i][-1]
        data_frame[i][:5] = data[i][:5]

        print(data[0])
        print(data[i])
        for j in zip(data_frame[0],data_frame[i]) : print(j)
        

        break
        # Rearrange table

    
     


def read_and_normalize():
    datafile = open(argv[1])
    data = list(map(lambda x : x.split(","), datafile.readlines()))

    remove_and_cast_features(data)
    
####################
read_and_normalize()
####################
