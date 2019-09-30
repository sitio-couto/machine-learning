import matplotlib.pyplot as plt 
import numpy as np
import neural as nr

def histogram(classes, amount=10):
    plt.hist(classes, bins='auto')
    plt.title("Amount of images per class in set")
    plt.xticks(range(amount))
    #plt.grid()
    plt.show()

def learning_curves(X, Y):
    plot_info = []

    X = X[:1024,:100]
    Y = Y[:,:100]
    feat = X.shape[0]
    out = Y.shape[0]

    batch = nr.Network([feat,feat,out], l=2)
    mini = nr.Network([feat,feat,out], l=2, T=batch.theta)
    stoch = nr.Network([feat,feat,out], l=2, T=batch.theta)

    b = batch.train(X, Y, type='b', e_lim=10, rate=0.05, analisys=True)
    m = mini.train(X, Y, type='m', e_lim=10, rate=0.05, mb_size=int(np.round(X.shape[1]*0.1)), analisys=True)
    s = stoch.train(X, Y, type='s', e_lim=10, rate=0.01,  analisys=True)

    train_data = [b,m,s]
    titles = ['Batch (rate 0.05)','Mini 10% (rate 0.05)','Stoch (rate 0.01)']

    for title,curve in zip(titles,train_data):
        C = []
        for T in curve.epochs_coef : C.append(nr.cost(X,Y,T))
        plot_info.append((title, range(0,curve.epochs_count+1), C))

    for (l,x,y) in plot_info : plt.plot(x, y, label=l)
    plt.title("Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()
    