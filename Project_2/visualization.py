import matplotlib.pyplot as plt
import random as rd
import numpy as np
import neural as nr
import itertools as it

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


def plot_confusion_matrix(confusion, classes, model='Neural Network'):
    '''
        Plots an already created confusion matrix for a generic amount of classes.
    '''
    
    fig, ax = plt.subplots(1)

    #Bounding box
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    plt.title('Confusion Matrix for ' + model)

    #Ticks
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thr = confusion.max()/2
    for i, j in it.product(range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(j, i, confusion[i, j],
            horizontalalignment='center',
            color='white' if confusion[i, j] > thr else 'black')

    plt.grid(False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.imshow(confusion, interpolation='nearest', cmap='Blues')
    plt.show()

    return fig, ax
