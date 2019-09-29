import matplotlib.pyplot as plt 
import numpy as np

def histogram(classes, amount=10):
    plt.hist(classes, bins='auto')
    plt.title("Amount of images per class in set")
    plt.xticks(range(amount))
    #plt.grid()
    plt.show()

def learning_curves(curves):
    for (l,x,y) in curves : plt.plot(x, y, label=l)
    plt.title("Learning Curves")
    plt.xlabel("Time")
    plt.ylabel("Cost")
    