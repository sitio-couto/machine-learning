import matplotlib.pyplot as plt
import random as rd
import numpy as np
import neural as nr

def histogram(classes, amount=10):
    plt.hist(classes, bins='auto')
    plt.title("Amount of images per class in set")
    plt.xticks(range(amount))
    #plt.grid()
    plt.show()

def learning_curves(X, Y, n=(0,3073), m=100):
    plot_info = []

    samples = rd.sample(range(Y.shape[1]),m)
    X = X[n[0]:n[1],samples]
    Y = Y[:,samples]
    feat = X.shape[0]
    out = Y.shape[0]

    l = 2
    t = 1800
    e = 50
    mbs = 0.1

    br = 0.05
    mr = 0.01
    sr = 0.001

    first_theta = nr.Network([feat,feat,out], l=l).theta

    methods = ['m']
    rates = [mr]
    titles = [f'Mini {100*mbs}% (rate {rates[0]})']

    for i,title in enumerate(titles):
        C = []
        model = nr.Network([feat,feat,out], T=first_theta, l=2)
        data = model.train(X, Y, type=methods[i], t_lim=t, e_lim=e, rate=rates[i], mb_size=int(np.ceil(X.shape[1]*mbs)), analisys=True)
        for T in data.epochs_coef : C.append(nr.cost(X,Y,T))
        plot_info.append((title, range(0,data.epochs_count+1), C))

    for (l,x,y) in plot_info : plt.plot(x, y, label=l)
    plt.title(f"Learning Curves (with {m} samples)")
    plt.xlabel(f"Epochs (limit of {t}s)")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()
    

def learning_with_history(history):
    '''
        Plots learning curves from history (dictionary of lists)
    '''
    keys = sorted(history.keys())
    for k in keys:
        plt.plot(history[k])
        
    plt.legend(keys, loc='upper left')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Learning Curve')
    plt.show()
