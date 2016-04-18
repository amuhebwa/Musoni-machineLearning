# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 21:30:53 2016

@author: AGGREY
"""
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

from pybrain.datasets import ClassificationDataSet

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()
x, y = iris.data, iris.target


input = 4 # x
target = 1 # y
classes = 3 # The output can be one of the three classes ie setisa, versicolour or virginica

ds = ClassificationDataSet(input,target, nb_classes=classes)
for i in range(len(x)):
    ds.addSample(x[i], y[i])

trndata_temp, partdata_temp = ds.splitWithProportion(0.60)

trndata = ClassificationDataSet(input,target, nb_classes=classes)
for n in xrange(0, trndata_temp.getLength()):
    trndata.addSample(trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1])


tstdata_temp, validata_temp = partdata_temp.splitWithProportion(0.5)

tstdata = ClassificationDataSet(input, target, nb_classes=classes)
for m in xrange(0, tstdata_temp.getLength()):
    tstdata.addSample(tstdata_temp.getSample(m)[0], tstdata_temp.getSample(m)[1])

validata = ClassificationDataSet(input, target, nb_classes = classes)
for j in xrange(0, validata_temp.getLength()):
    validata.addSample(validata_temp.getSample(j)[0], validata_temp.getSample(j)[1])

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()
validata._convertToOneOfMany()

# TIME TO CREATE A FNN NEURAL NETWORK WITH 4 INPUTS, 3 HIDDEN NEURONS AND 3 OUTPUTS 
FNN_INPUT = 4
FNN_HIDDEN = 5
FNN_OUTPUT = 3
net = buildNetwork(FNN_INPUT, FNN_HIDDEN, FNN_OUTPUT, outclass = SoftmaxLayer)

# set up training for back propagation algorithm
# Notice that we set up using the train data
trainer = BackpropTrainer(net, dataset = trndata, momentum = 0.1, verbose = True, weightdecay = 0.01)
# Now, lets train the neural network for just 50 epochs while plotting the training error
#trnerr, valerr = trainer.trainUntilConvergence(dataset=trndata, maxEpochs = 50)
#plt.plot(trnerr, 'b', valerr, 'r')
trainer.trainOnDataset(trndata, 1000)
#print('Total Epochs',trainer.totalepochs)

# Evaluate the error rate on test data

#out = net.activateOnDataset(tstdata).argmax(axis=1)
#pe = percentError(out, tstdata['class'])
#print('Percent Error :', pe)
#At this point, the model only learnt the training data and tested on the test data

# ------------------------------------------------#
# To predict how the model works when seeing new dataset, rather than memorizing,
#We run  our neural network  model to predict on the final validation dataset

out = net.activateOnDataset(tstdata)
out = out.argmax(axis=1)


# ... and test on the validation error
output = np.array([net.activate(k) for k, _ in validata])
output = output.argmax(axis=1)
print(output)
print('------------------------------------------------')
zz = percentError(output, validata['class'])
print 'Percent Error', zz
print(validata['class'])