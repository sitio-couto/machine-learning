from random import randint
import numpy as np

import new_model as model
import normalization as norm
import gradient_descent as desc
import visualization as graph

# ### Getting Training Set ###
# Returns the data split between features and target.
X, Y, feat_list = model.prepare_dataset("Datasets/training.csv")
# Normalize the features pointed by the received model
X = norm.normalize_data(X, choice=1, features=feat_list)
# Cast, add bias and randomize initial Thetas
X, T, Y = desc.numpy_and_bias(X, Y)

# Validation set
# features to be normalized are the same
X_val, Y_val, _ = model.prepare_dataset("Datasets/validate.csv")
X_val = norm.normalize_data(X_val, choice=1, features=feat_list)
X_val, _, Y_val = desc.numpy_and_bias(X_val, Y_val)

#T = desc.descent(X, T, Y, t_lim=180, e_lim=10000, type="s")
#print("TRAIN SCORE=>", desc.score(X, T, Y))
#print("VALID SCORE=>", desc.score(X_val, T, Y_val))
# ### FITTING EVALUATION ###
#T = desc.descent(X, T, Y, t_lim=30, e_lim=1000)
#print(desc.epochs_count)

# # Check some random predictions accuracy
#for i in range(20):
#    x = randint(0,Y_val.shape[0])
#    print(int(desc.predict(X_val[[x]], T)), "=>", int(Y_val[[x]]))

#graph.learning_curve(X, Y, X_val, Y_val, desc.epochs_info[0], desc.cost)

# ### NORMAL EQUATION ###
T = desc.normal_equation(X,Y)
# # Results
print("Train Normal Score =>", desc.score(X,T,Y))

# # Validate
print("Validation Normal Score =>", desc.score(X_val, T, Y_val))

# # Check some random predictions accuracy
#for i in range(20):
#    x = randint(0,Y_val.shape[0])
#    print(int(desc.predict(X_val[[x]], T)), "=>", int(Y_val[[x]]))

# ### SKLEARN ###
# # Train coeficients
Y = np.ravel(Y)
clf = desc.sk_regressor(X, Y)
# # Training results
print("Training Score =>", clf.score(X,Y))

# # Validate
Y_val = np.ravel(Y_val)
print("Validation Score=>", clf.score(X_val,Y_val))

# # Check some random predictions accuracy
for i in range(20):
    x = randint(0,Y_val.shape[0])
    print(int(clf.predict(X_val[[x]])), "=>", int(Y_val[[x]]))

# ### TESTING GRADIENTS ###
# # Returns the data without the header
# X, Y, feat_list = model.prepare_dataset("Datasets/training.csv")
# # Normalize the features pointed by the received model
# X = norm.normalize_data(X, choice=1, features=feat_list)
# # Cast, add bias and randomize initial Thetas
# X, T, Y = desc.numpy_and_bias(X, Y)

# desc.descent(X, T, Y, t_lim=30, e_lim=10000)
# batch = []
# batch.append([desc.cost(X, i, Y)/10**6 for i in desc.epochs_info[0]])
# batch.append(desc.epochs_info[1][:])

# desc.descent(X, T, Y, t_lim=30, e_lim=10000, type='m', mb_size=int(0.05*Y.shape[0]))
# minib = []
# minib.append([desc.cost(X, i, Y)/10**6 for i in desc.epochs_info[0]])
# minib.append(desc.epochs_info[1][:])

# desc.descent(X, T, Y, t_lim=30, e_lim=10000, type='s')
# stoch = []
# stoch.append([desc.cost(X, i, Y)/10**6 for i in desc.epochs_info[0]])
# stoch.append(desc.epochs_info[1][:])

# graph.gradient_comparison(batch, stoch, minib)
