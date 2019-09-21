import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import re

from scipy import stats

import first_model as model
import normalization as norm
import gradient_descent as desc

def stats_hist(data):
    ''' Transforms data to pandas dataframe and gets stats/histogram
    '''
    pd.DataFrame(data)
    print(data.describe())
    hist=data.hist(bins=50)
    plt.show()

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
            traffic_hour.append(np.array([0]*24))
        traffic_hour[-1][hour] = x[-1]

    avg = sum(traffic_hour)/len(traffic_hour)
    plt.plot(range(0,24), avg)
    plt.xticks(range(0,24))
    plt.title("Análise do tráfego por hora")
    plt.xlabel("Horas do dia (00h-24h)")
    plt.ylabel("Média diária de tráfego")
    print('Plotting')
    plt.grid()
    plt.show()

    return

def avg_traffic_day_weekly(data):
    # NEEDS HEADER
    # OBS: there are duplicated time stamps to separate more than one
    # enviromental condition (p.e. if its foggy and cloudy there will be duplicates)
    traffic_day = []
    data = pd.DataFrame(data)
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['weekday'] = data['date_time'].dt.weekday
    
    for i in range(0,7):
        dt = data.loc[data['date_time'].dt.weekday == i]
        traffic_day.append(dt['traffic_volume'].mean())
        
    plt.plot(range(0,7), traffic_day)
    plt.xticks(range(0,7))
    plt.title("Análise do tráfego por dia")
    plt.xlabel("Dias da semana (Segunda a Domingo)")
    plt.ylabel("Média semanal de tráfego")
    print('Plotting')
    plt.grid()
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
    desc_main_avg = np.array(sums)/np.array(counts)

    counts = [x[0] if x[0]>0 else 1  for x in desc.values()]
    sums = [x[1] for x in desc.values()]
    desc_avg = np.array(sums)/np.array(counts)
    
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
  
    plt.grid()
    plt.show()
    input()
    
    return


#### MODEL ANALYSIS ####

def learning_curve():
    X, Y, feat_list = model.prepare_dataset("Datasets/training.csv")
    X = norm.normalize_data(X, choice=1, features=feat_list)
    X, o, Y = desc.numpy_and_bias(X, Y)
    desc.descent(X, o, Y, t_lim=30, e_lim=200, rate=1)
    
    Xv, Yv, feat_list = model.prepare_dataset("Datasets/validate.csv")
    Xv = norm.normalize_data(Xv, choice=1, features=feat_list)
    Xv, _, Yv = desc.numpy_and_bias(Xv, Yv)

    train = []
    valid = []
    T = desc.epochs_info[0]
    
    step = int(max(1, np.floor(len(T)/100)))
    exp = range(0, len(T), step)

    for i in exp:
        train.append(desc.cost(X,T[i],Y)/10**6)
        valid.append(desc.cost(Xv,T[i],Yv)/10**6)

    plt.plot(exp, valid, label='Validation')
    plt.plot(exp, train, label='Training', ls='--')
    plt.legend()
    plt.xlabel('Experience (Epochs)')
    plt.ylabel('Learning (x10^6)')
    plt.title('Learning Curve')
    plt.grid()
    plt.show()
    return 


def gradient_comparison(batch, stoch, minib):
    plt.plot(batch[1], batch[0], label='batch ('+str(len(batch[0])-1)+' Epochs)')
    plt.plot(stoch[1], stoch[0], label='stochastic ('+str(len(stoch[0])-1)+' Epochs)')
    plt.plot(minib[1], minib[0], label='mini batch 2% ('+str(len(minib[0])-1)+' Epochs)')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Cost (x10^6)')
    plt.title('Gradients Curve')
    plt.show()
    return 

def gradient_comparison(alpha):
    
    X, Y, feat_list = model.prepare_dataset("Datasets/training.csv")
    X = norm.normalize_data(X, choice=1, features=feat_list)
    X, T, Y = desc.numpy_and_bias(X, Y)

    desc.descent(X, T, Y, t_lim=30, e_lim=10000, rate=alpha)
    batch = []
    batch.append([desc.cost(X, i, Y)/10**6 for i in desc.epochs_info[0]])
    batch.append(desc.epochs_info[1][:])

    desc.descent(X, T, Y, t_lim=30, e_lim=10000, type='m', rate=alpha, mb_size=int(0.05*Y.shape[0]))
    minib = []
    minib.append([desc.cost(X, i, Y)/10**6 for i in desc.epochs_info[0]])
    minib.append(desc.epochs_info[1][:])

    desc.descent(X, T, Y, t_lim=30, e_lim=10000, rate=alpha, type='s')
    stoch = []
    stoch.append([desc.cost(X, i, Y)/10**6 for i in desc.epochs_info[0]])
    stoch.append(desc.epochs_info[1][:])
   
    plt.plot(batch[1], batch[0], label='batch ('+str(len(batch[0])-1)+' Epochs)')
    plt.plot(stoch[1], stoch[0], label='stochastic ('+str(len(stoch[0])-1)+' Epochs)')
    plt.plot(minib[1], minib[0], label='mini batch 2% ('+str(len(minib[0])-1)+' Epochs)')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Cost (x10^6)')
    plt.title('Gradients Curve (alpha = '+str(alpha)+')')
    plt.show()
    return 

def best_alpha_gradient_comparison():
    
    X, Y, feat_list = model.prepare_dataset("Datasets/training.csv")
    X = norm.normalize_data(X, choice=1, features=feat_list)
    X, T, Y = desc.numpy_and_bias(X, Y)

    desc.descent(X, T, Y, t_lim=30, e_lim=10000, rate=1)
    batch = []
    batch.append([desc.cost(X, i, Y)/10**6 for i in desc.epochs_info[0]])
    batch.append(desc.epochs_info[1][:])

    desc.descent(X, T, Y, t_lim=30, e_lim=10000, type='m', rate=0.9, mb_size=int(0.05*Y.shape[0]))
    minib = []
    minib.append([desc.cost(X, i, Y)/10**6 for i in desc.epochs_info[0]])
    minib.append(desc.epochs_info[1][:])

    desc.descent(X, T, Y, t_lim=30, e_lim=10000, rate=0.01, type='s')
    stoch = []
    stoch.append([desc.cost(X, i, Y)/10**6 for i in desc.epochs_info[0]])
    stoch.append(desc.epochs_info[1][:])
   
    plt.plot(batch[1], batch[0], label='batch ('+str(len(batch[0])-1)+' Epochs & alpha = 1)')
    plt.plot(stoch[1], stoch[0], label='stochastic ('+str(len(stoch[0])-1)+' Epochs & alpha = 0.01)')
    plt.plot(minib[1], minib[0], label='mini batch 2% ('+str(len(minib[0])-1)+' Epochs & alpha = 0.9)')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Cost (x10^6)')
    plt.title('Gradients Curve')
    plt.show()
    return 

def alpha_comparison(type, title, alphas):
    
    X, Y, feat_list = model.prepare_dataset("Datasets/training.csv")
    X = norm.normalize_data(X, choice=1, features=feat_list)
    X, T, Y = desc.numpy_and_bias(X, Y)

    for alpha in alphas:
        desc.descent(X, T, Y, t_lim=30, e_lim=10000, rate=alpha, type=type, mb_size=int(0.02*Y.shape[0]))
        y = [desc.cost(X, i, Y)/10**6 for i in desc.epochs_info[0]]
        x = desc.epochs_info[1]
        plt.plot(x, y, label=str(alpha))

    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Cost (x10^6)')
    plt.title(title)
    plt.show()
    return 

def norm_check():
    X, Y, feat_list = model.prepare_dataset("Datasets/training.csv")
    Xa = np.array(norm.normalize_data(X, choice=1, features=feat_list))
    Xb = np.array(norm.normalize_data(X, choice=2, features=feat_list))
    Xc = np.array(norm.normalize_data(X, choice=3, features=feat_list))
    
    for x in [Xa,Xb,Xc]:
        print("--- Normalizacao ---")
        for j in feat_list:
            print("-- Feature "+str(j)+":")
            print("Max=>",np.amax(x[:,[j]]))
            print("Media=>",np.mean(x[:,[j]]))
            print("Min=>",np.amin(x[:,[j]]))
            print("Stdv=>",np.std(x[:,[j]]))
            print("Moda=>",stats.mode(x[:,[j]])[0])



##### CALLING AREA #####

# alpha_comparison('s',"Stochastic Learning Rates", [0.1, 0.01, 0.001,0.0001])
# gradient_comparison(0.01)
# learning_curve()
# norm_check()

