# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:13:32 2017

@author: James

Deep Softmax Neural Network program to classify a Landsat 7 ETM+ satellite image of SLC

Input for X:
8-band satellite image converted to numpy array, bands (1,2,3,4,5,6low,6high,7)
produced using geotif_to_numpy_array.py in QGIS
Dimensions are (bands, rows, cols)

Input for Y:
classified satellite image, with 6 classes/features, produced using unsupervised
classification (K-means) in QGIS  
dimensions are (rows, cols)  

Accuracy for train data is 99.7% after only 3 iterations
Accuracy for test data is also 99.7%

NOTE: Images are large and neural network runs slow, but 5 iterations only takes
a few minutes and gives good estimate of network accuracy

After classifying train and test data, the model is saved to disk, re-opened,
and applied to a different Landsat 7 ETM+ image for classification.
The predictions on the new image are turned into an array and saved to disk
so they can be converted into a thematic geotiff and viewed in QGIS.

"""

from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

#################### OPEN AND PRE-PROCESS CLASSIFIED IMAGES ####################
# classified image is a single band, with values ranging from 0 to 5,
# indicating which of the 6 classes each pixel is assigned

# open classified tiff image of SLC North and SLC South
im_Y = Image.open('C:/users/james/desktop/Satellite Data/LE07_20010531/SLC_test.tif')

# convert tiff to numpy array
Y = np.array(im_Y)
del im_Y  # delete to clear memory

# Reshape/flatten the classified image 
# The "-1" makes reshape flatten the remaining dimensions (the column dimension becomes 1)
# each image is flattend to 1 column, many rows
flat_length_tst = Y.shape[0]*Y.shape[1]
Y_flat = np.reshape(Y, (flat_length_tst,-1))
del Y  # delete to clear memory

# squeeze array so it is of shape (n_samples, ), removing the column index
Y_flat = np.squeeze(Y_flat)

print('Shape of Y_flat',Y_flat.shape)
print()
#################### OPEN AND PRE-PROCESS MULTI-BAND IMAGES ###################

# open 8-band numpy array
# this array was produced using the geotiff_to_array.py program in QGIS
X = np.load('C:/users/james/desktop/Satellite Data/LE07_20010531/test_array.npy')

 # convert uint array into int32 array,
X = np.int32(X)   # so they can be squeezed

# reshape array to have dim (n_features, n_samples, 1)
X = np.reshape(X, (X.shape[0],X.shape[1]*X.shape[2],-1))

# squeeze training array to remove column index, will have dim (n_features, n_samples)
X = np.squeeze(X)

# transpose array to have dim (n_samples, n_features)
X = X.T

print('Shape of X',X.shape)
print()

# multiple hidden layer networks are sensitive to feature scaling.
# standardize X data to have 0 mean and variance of 1.
# NOTE: THIS STEP HELPS IMPROVE COST/LOSS AND ACCURACY !!

from sklearn.preprocessing import StandardScaler
print('Normalizing the data....be patient')
scaler = StandardScaler()
scaler.fit(X)  # get standardization parameters, based on data
X = scaler.transform(X)  # apply standardization parameters to data

print('Mean of X_test:',np.mean(X))  # should be near 0
print('Variance of X_test:',np.var(X)) # should be 1

############## SHUFFLE AND SPLIT DATA INTO TEST AND TRAIN SETS ################
print()
print('Shuffling data and splitting into train and test sets....be patient')

# create list of randomly selected numbers from 0 to X.shape[0]
permutation = list(np.random.permutation(X.shape[0]))
shuffled_X = X[permutation,:]  # apply permutation to X to shuffle it
shuffled_Y = Y_flat[permutation,]  # apply permutation to Y to shuffle it
del X  # delete to clear memory
del Y_flat  # delete to clear memory

X_train = shuffled_X[::2,:] # even indices become X_train
X_test = shuffled_X[1::2,:] # odd indices become X_test
del shuffled_X  # delete to clear memory

Y_train = shuffled_Y[::2,] # even indices become Y_train
Y_test = shuffled_Y[1::2,] # odd indices become T_test
del shuffled_Y  # delete to clear memory

############# DEFINE THE MULTI-LAYER MODEL ARCHITECTURE #######################
""" Default parameters:
activation='relu'   for hidden layers, others include 'logistic' for sigmoid, and 'tanh'
alpha=1e-05         L2 Regularization term to avoid overfitting
batch_size='auto'   batch size will = min(200,n_samples)
beta_1=0.9          Momentum term to accelerate gradient descent, part of Adam Optimization
beta_2=0.999        RMS Prop term to accelerate gradient descent, part of Adam Optimization
early_stopping=False When True, training stops when loss<tol for 2  consecutive epochs
epsilon=1e-08       Prevents division by zero, part of Adam Optimization
hidden_layer_sizes=(15,) A tuple listing number of neurons for each hidden layer,
                        excludes input layer, but includes output layer.
learning_rate='constant' Only used when solver='sgd'
                     'invscaling' decreases learning rate by power_t, 'adaptive'
                     keeps rate constant while loss decreases, then divides by 5
                     if loss is not > tol 
learning_rate_init=0.001    Starting learning rate
max_iter=200        Number of iterations, or when convergence is reached (tol)
momentum=0.9        Momentum term, only for 'sgd'
nesterovs_momentum=True Only for 'sgd'
power_t=0.5        exponent for inverse scaling learning rate, only for 'sgd'
random_state=1     random seed
shuffle=True       randomly shuffle mini-batches
solver='adam'       other options are 'lbfgs', 'sgd' (stochastic gradient descent)
tol=0.0001          training stops when loss changes by less than this value
validation_fraction=0.1     fraction of training data reserved for model validation
                            for early stopping
verbose=False       True, prints progress
warm_start=False    Set to True and max_iter=1 if you wish to put clf.fit(X,Y) in a for-loop
                    for more control. Reuse previous parameters as initialization
"""

# DEFINE MODEL ARCHITECTURE:
    
clf = MLPClassifier(activation='relu',
                    hidden_layer_sizes=(8,6,6),
                    learning_rate_init=0.01,
                    max_iter=1,
                    solver='adam',
                    warm_start=True)

print()
################# FIT THE MODEL TO THE DATA ###################################
# X: array of shape(n_samples, n_features)
# Y: array of shape(n_samples)
# putting clf.fit(X,Y) in a for-loop allows more control over model, such
# as collecting costs for plotting...
# Use warm_start=True and max_iter=1 when using clf.fit in a loop

cost = []
for iter in range(11):
    clf.fit(X_train,Y_train)
    
    if iter%1==0:
        print('Loss after iteration', iter,':', clf.loss_)
        cost.append(clf.loss_)

############# PREDICTIONS AND ACCURACY ########################################

# TRAIN PREDICTIONS:
train_preds = clf.predict((X_train)) # make predictions using model 

# use built-in method (.score) to calc accuracy
print()
print('TRAIN SET ACCURACY: %f' % clf.score(X_train,Y_train)) # use built-in function to calc accuracy


# TEST PREDICTIONS:
preds = clf.predict((X_test)) # make predictions using model 

# use built-in method (.score) to calc accuracy
print()
print('TEST SET ACCURACY: %f' % clf.score(X_test,Y_test)) # use built-in function to calc accuracy

############## GET MODEL W,b PARAMETERS #######################################

#print()
#print('MODEL WEIGHTS')
#print(clf.coefs_) # print all W  values for all layers
#print()
#print('MODEL BIASES')
#print(clf.intercepts_) # print biases, b values for all layers
print()
print('SHAPE OF LAYERS IN NEURAL NETWORK')
for coef in clf.coefs_:
    print(coef.shape)  # print shape (dim) of each W array


############### OUTPUT LAYER PROBABILITIES ####################################
# each row should sum to 1
# column index of max value of each row is class value

#print()
#print('CLASS PROBABILITIES')
#print(clf.predict_proba(X_test))

############### LOSS/COST ####################################################

print()
print('CURRENT LOSS')
print(clf.loss_)  # current loss

plt.figure(figsize=(6,4))
plt.plot(np.squeeze(cost))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Softmax classifier performance")
plt.show()

################ SAVE THE MODEL DISK ##########################################

import pickle
filename = 'C:/users/james/desktop/Satellite Data/LE07_20010531/Image_classifier_model.sav'
pickle.dump(clf, open(filename, 'wb'))



############## OPEN DIFFERENT SATELLITE IMAGE FOR CLASSIFICATION ##############
############## AND PRE-PROCESS THE IMAGE BEFORE USE IN MODEL ##################

# open 8-band numpy array
# this array was produced using the geotiff_to_array.py program in QGIS
X_next = np.load('C:/users/james/desktop/Satellite Data/LE07_20020502/test_array.npy')

 # convert uint array into int32 array,
X_next = np.int32(X_next)   # so they can be squeezed
print('Shape of X_next prior to flattening',X_next.shape)

bands = X_next.shape[0]
rows = X_next.shape[1]
cols = X_next.shape[2]

# reshape array to have dim (n_features, n_samples, 1)
X_next = np.reshape(X_next, (X_next.shape[0],X_next.shape[1]*X_next.shape[2],-1))

# squeeze training array to remove column index, will have dim (n_features, n_samples)
X_next = np.squeeze(X_next)

# transpose array to have dim (n_samples, n_features)
X_next = X_next.T

print('Shape of X_next after flattening',X_next.shape)
print()

# multiple hidden layer networks are sensitive to feature scaling.
# standardize X data to have 0 mean and variance of 1.
# NOTE: THIS STEP HELPS IMPROVE COST/LOSS AND ACCURACY !!

#from sklearn.preprocessing import StandardScaler
print('Normalizing the data....be patient')
scaler = StandardScaler()
scaler.fit(X_next)  # get standardization parameters, based on data
X_next = scaler.transform(X_next)  # apply standardization parameters to data

print('Mean of X_test:',np.mean(X_next))  # should be near 0
print('Variance of X_test:',np.var(X_next)) # should be 1

################# RELOAD MODEL AND RUN ON DIFFERENT SATELLITE IMAGE ###########

# re-load the model from disk, filename is defined above when saving the model
reload_clf = pickle.load(open(filename, 'rb'))

# make class predictions on different satellite image using model
new_preds = reload_clf.predict((X_next))

################### CREATE THEMATIC IMAGE USING PREDICTIONS ####################

# reshape array back to original dimensions of classified image
classified_arr = np.reshape(new_preds, (rows,cols))

# save array to disk
np.save('C:/users/james/desktop/Satellite Data/LE07_20020502/classified_array',classified_arr)

# NOTE: USE ARRAY_TO_GEOTIFF.PY PROGRAM IN QGIS TO CONVERT THE SAVED ARRAY 
# INTO A GEOTIFF TO BE VIEWED IN QGIS.




















