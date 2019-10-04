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
    s=50
    
    func = ['sg','sm']
    titles = ['Sigmoid']

    for i,title in enumerate(titles):
        C = []
        model = nr.Network([feat,feat,out],  f=func[i], seed=23)
        data = model.train(X, Y, type='m', t_lim=t, e_lim=e, rate=r, mb_size=b, sampling=s)
        for T in data.coef : C.append(nr.accuracy(X,Y,T))
        plot_info.append((title, range(0,len(data.coef)), C))

    for (l,x,y) in plot_info : plt.plot(x, y, label=l)
    plt.title(f"Learning Curve \n {m} samples | rate {r} | {t} sec | batch {b}")
    plt.xlabel(f"Iterations (x{s})")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    

def second_model(X, Y, Xv, Yv, n=(0,3073), m=1000):
    plot_info = []

    samples = rd.sample(range(Y.shape[1]), m)
    X = X[n[0]:n[1],samples]
    Y = Y[:,samples]
    feat = X.shape[0]
    out = Y.shape[0]

    e=50
    t=300
    r=0.005
    b=256
    s=40
    
    arc1 = [int(np.floor(1024/2**i)) for i in range(5)] + [32,32,10]
    arc2 = [1024,1024,10]
    arc = [arc1, arc2]

    func = ['sg','sg']
    titles = ['First Model', 'Second Model']

    for i,title in enumerate(titles):
        C = []
        print("Architecture:", arc[i])
        model = nr.Network([feat,feat,out],  f=func[i], seed=23)
        data = model.train(X, Y, type='m', t_lim=t, e_lim=e, rate=r, mb_size=b, sampling=s)
        for T in data.coef : C.append(nr.accuracy(X,Y,T))
        plot_info.append((title, range(0,len(data.coef)), C))

    for (l,x,y) in plot_info : plt.plot(x, y, label=l)
    plt.title(f"Learning Curve \n {m} samples | rate {r} | {t} sec | batch {b}")
    plt.xlabel(f"Iterations (x{s})")
    plt.ylabel("Accuracy")
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
