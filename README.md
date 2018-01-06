# Deep-Softmax-Neural-Network

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
