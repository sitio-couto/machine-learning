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
    

def comparing_models(X, Y, Xv, Yv, n=(0,3073), m=80000):
    plot_info = []

    samples = rd.sample(range(Y.shape[1]), m)
    X = X[n[0]:n[1],samples]
    Y = Y[:,samples]
    feat = X.shape[0]
    out = Y.shape[0]

    e=50
    t=30
    r=0.005
    b=256
    s=10
    
    # arc1 = [3072, 3072, 10]
    # arc4 = [3072, 768, 192, 48, 10]
    arc5 = [3072, 256, 128, 10]
    arc6 = [3072, 128, 64, 32, 10]
    arc = [arc5, arc6]
    titles = ['Fifth Model','Sixth Model']

    for i,title in enumerate(titles):
        C = []
        print("Architecture:", arc[i])
        model = nr.Network(arc[i], seed=23)
        data = model.train(X, Y, type='m', t_lim=t, e_lim=e, rate=r, mb_size=b, sampling=s)
        for T in data.coef : C.append(nr.accuracy(X,Y,T))
        plot_info.append((title, range(0,len(data.coef)), C))

    for (l,x,y) in plot_info : plt.plot(x, y, label=l)
    plt.title(f"Learning Curve \n {m} samples | rate {r} | {t} sec | batch {b}")
    plt.xlabel(f"Iterations (x{s})")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def overfitting(X, Y, Xv, Yv, n=(0,3073), m=1000):
    plot_info = []

    samples = rd.sample(range(Y.shape[1]), m)
    X = X[n[0]:n[1],samples]
    Y = Y[:,samples]
    feat = X.shape[0]
    out = Y.shape[0]

    e=50
    t=600
    r=0.005
    b=256
    s=50
    
    arc = [3072, 256, 128, 10]

    C = []
    V = []
    print("Architecture:", arc)
    model = nr.Network(arc, seed=23)
    data = model.train(X, Y, type='m', t_lim=t, e_lim=e, rate=r, mb_size=b, sampling=s)
    for T in data.coef : 
        C.append(nr.accuracy(X,Y,T))
        V.append(nr.accuracy(Xv,Yv,T)) 
    plot_info.append(('Training Set', range(0,len(data.coef)), C))
    plot_info.append(('Validation Set', range(0,len(data.coef)), V))


    for (l,x,y) in plot_info : plt.plot(x, y, label=l)
    plt.title(f"Learning Curve \n {m} samples | rate {r} | {t} sec | batch {b}")
    plt.xlabel(f"Iterations (x{s})")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def optimization(X, Y, Xv, Yv, m=80000):
    plot_info = []

    samples = rd.sample(range(Y.shape[1]), m)
    X = X[:,samples]
    Y = Y[:,samples]
    feat = X.shape[0]
    out = Y.shape[0]

    # Set constant hyperparameters
    e=50
    t=40
    r=0.005
    b=1024
    s=10
    
    # Variable aspects/hyperparams per plot
    opt = ['adadelta', None]
    title = ['Adadelta', 'Vanilla']
    arc = [3072, 256, 128, 10]

    print("Architecture:", arc)
    for i in range(len(opt)):
        C = []
        model = nr.Network(arc, seed=23)
        data = model.train(X, Y, 
                            opt=opt[i], 
                            type='m', 
                            t_lim=t, 
                            e_lim=e, 
                            rate=r, 
                            mb_size=b, 
                            sampling=s)
        for T in data.coef : C.append(nr.accuracy(X,Y,T))
        plot_info.append((title[i], range(0,len(data.coef)), C))
    # plot_info.append(('Validation Set', range(0,len(data.coef)), V))


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
