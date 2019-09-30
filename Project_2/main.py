# MC886 - Machine Learning - UNICAMP
# Project 2 - Logistic Regression and Neural Networks
#
# Authors:
# Victor Ferreira Ferrari - RA 187890
# Vinicius Couto Espindola- RA 188115

import numpy as np
import normalization as norm
import visualization as vis
import logistic as lr
#import neural as nr
import misc

# Getting Sets
train = np.load('Dataset/train.npz')
valid = np.load('Dataset/val.npz')
X, Y  = train['xs'].astype('float32') , train['ys'].astype('int8')
X_v, Y_v = valid['xs'].astype('float32') , valid['ys'].astype('int8')
print("Dataset read!")

# Visualization
#vis.histogram(Y,10)

# Normalization
choice = 1
stats = norm.get_stats(X, choice)
X = norm.normalize_data(X, stats, choice)
print("Training data Normalized!")
X_v = norm.normalize_data(X_v, stats, choice)
print("Val Data normalized!")

##### NEURAL NETWORK ####
#print("Starting neural network section")
# Adjusting input matrices (assumes X is normalized)
#Xn = X.T
#Yn = norm.out_layers(Y)
#print("Handled input")
# Builds network object
#feat = Xn.shape[0]
#out = Yn.shape[0]
#batch_size = int(np.round(Xn.shape[1]*0.1))
#model = nr.Network([feat,feat,out], l=2)
#print('Created model')
#print("Initial Cost:", model.cost(Xn, Yn))
#exit(1)
# Train model
#data = model.train(Xn, Yn, type='m', mb_size=batch_size)
#print("Trained Cost:", model.cost(Xn, Yn))

# # Neural Network descent visualization
# vis.learning_curves(Xn, Yn, m=80000)

##### MULTINOMIAL LOGISTIC REGRESSION ####
# Initial coefficients and bias.
X = np.insert(X, 0, 1, axis=1)
X_v = np.insert(X_v, 0, 1, axis=1)
print("Bias Added")
classes = np.max(Y) + 1
T = misc.init_coefs(X.shape[1], classes, 57).astype('float32')

# Logistic Regression (Softmax)
print("Regression:")
T = lr.gradient_descent(X, Y, X_v, Y_v, T, 0.01, 200)
v_pred = lr.predict(X_v, T)
confusion = misc.confusion_matrix(Y_v, v_pred, classes)
acc = misc.accuracy(confusion)
print(acc)

