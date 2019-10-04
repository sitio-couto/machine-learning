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

def sigmoid_vs_softmax(X, Y, Xv, Yv, n=(0,3073), m=1000):
    plot_info = []

    samples = rd.sample(range(Y.shape[1]), m)
    X = X[n[0]:n[1],samples]
    Y = Y[:,samples]
    feat = X.shape[0]
    out = Y.shape[0]

    e=50
    t=3600
    r=0.01
    b=256
    s = 50
    
    func = ['lg','sm']
    titles = ['Sigmoid','Softmax']

    for i,title in enumerate(titles):
        C = []
        model = nr.Network([feat,feat,out],  f=func[i], seed=23)
        data = model.train(X, Y, type='m', t_lim=t, e_lim=e, rate=r, mb_size=b, sampling=s)
        for T in data.coef : C.append(nr.accuracy(X,Y,T))
        plot_info.append((title, range(0,len(data.coef)), C))

    for (l,x,y) in plot_info : plt.plot(x, y, label=l)
    plt.title(f"Learning Curve \n {m} samples | rate 0.01 | 300 sec | batch 256")
    plt.xlabel(f"Iterations (x{s})")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    