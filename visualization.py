import matplotlib.pyplot as plt 
<<<<<<< HEAD
# from matplotlib import show, draw
from numpy import array, floor
=======
import pandas as pd
import numpy as np
>>>>>>> d84d01e0c20b2ba78d50f6613694bc05e9210ab4
import re

def stats(data):
	''' Transforms data to pandas dataframe and gets stats
	'''
	pd.DataFrame(data)
	print(data.describe())

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
<<<<<<< HEAD
            traffic_hour.append(array([0]*24))
=======
            traffic_hour.append(np.array([0]*24))
>>>>>>> d84d01e0c20b2ba78d50f6613694bc05e9210ab4
        traffic_hour[-1][hour] = x[-1]

    avg = sum(traffic_hour)/len(traffic_hour)
    plt.plot(range(0,24), avg)
    plt.title("Análise do tráfego por hora")
    plt.xlabel("Horas do dia (00h-24h)")
    plt.ylabel("Média diária de tráfego")
<<<<<<< HEAD
    print('Ploting')
=======
    plt.grid()
>>>>>>> d84d01e0c20b2ba78d50f6613694bc05e9210ab4
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
<<<<<<< HEAD
    desc_main_avg = array(sums)/array(counts)

    counts = [x[0] if x[0]>0 else 1  for x in desc.values()]
    sums = [x[1] for x in desc.values()]
    desc_avg = array(sums)/array(counts)
=======
    desc_main_avg = np.array(sums)/np.array(counts)

    counts = [x[0] if x[0]>0 else 1  for x in desc.values()]
    sums = [x[1] for x in desc.values()]
    desc_avg = np.array(sums)/np.array(counts)
>>>>>>> d84d01e0c20b2ba78d50f6613694bc05e9210ab4
    
    a = plt.figure(1)
    plt.plot(list(desc.keys()), desc_avg)
    plt.xticks(list(desc.keys()), rotation='vertical')
    plt.margins(0)
    plt.subplots_adjust(bottom=0.5)
    plt.title("Análise do tráfego por descrição específica do clima")
    plt.xlabel("Descrições específicas do clima")
    plt.ylabel("Média diária de tráfego")
    plt.grid()
    
    b = plt.figure(2)
    plt.plot(list(desc_main.keys()), desc_main_avg)
    plt.title("Análise do tráfego por descrição geral do clima")
    plt.xlabel("Descrições gerais do clima")
    plt.ylabel("Média diária de tráfego")
  
<<<<<<< HEAD
    plt.draw()
=======
    plt.grid()
    plt.show()
>>>>>>> d84d01e0c20b2ba78d50f6613694bc05e9210ab4
    input()
    
    return


#### MODEL ANALYSIS ####

def learning_curve(X, Y, Xv, Yv, knowledge, test):
    train = []
    valid = []
    T = knowledge
    step = int(max(1, floor(len(T)/100)))
    exp = range(0, len(T), step)

    for i in exp:
        train.append(test(*(X,T[i],Y))/1000)
        valid.append(test(*(Xv,T[i],Yv))/1000)

    plt.plot(exp, train, label='Training')
    plt.plot(exp, valid, label='Validation')
    plt.xlabel('Experience')
    plt.ylabel('Learning (x10^3)')
    plt.title('Learning Curve')
    plt.show()
    return 