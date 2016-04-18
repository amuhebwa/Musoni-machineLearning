# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:55:21 2016

@author: AGGREY
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.datasets import ClassificationDataSet


class LoadDataset(object):
    """Loads dataset from CSV file"""
    
    def __init__(self):
        pass
    
    def load_file(self, file_name, required_columns=None):
        if(required_columns is None):
            dataset = pd.read_csv(file_name, low_memory=False)
        else:
            dataset = pd.read_csv(file_name, usecols=required_columns, low_memory=False)       
        return dataset
    
file_name = 'groups_final_dataset.csv'
df = LoadDataset().load_file(file_name,required_columns = None)
data = df.drop(['Average Over Due Days'], axis=1)

# Randomize the data set
data = data.sample(frac=1).reset_index(drop=True)

# Devide the data set into input->target
x = data[['Number Of Loans','Average Principal']]
y = data[['Low/High Risk']]

x = np.asarray(x)
y = np.asarray(y)

input = 2 # x
target = 1 # y
classes = 2 # The output can be one of the two classes, Not Risky(0) OR Risky(1)

ds = ClassificationDataSet(input,target, nb_classes=classes)
for i in range(len(x)):
    ds.addSample(x[i], y[i])


trndata_temp, partdata_temp = ds.splitWithProportion(0.60)
trndata = ClassificationDataSet(input,target, nb_classes=classes)
for n in xrange(0, trndata_temp.getLength()):
    trndata.addSample(trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1])

tstdata_temp, validata_temp = partdata_temp.splitWithProportion(0.9)

tstdata = ClassificationDataSet(input, target, nb_classes=classes)
for m in xrange(0, tstdata_temp.getLength()):
    tstdata.addSample(tstdata_temp.getSample(m)[0], tstdata_temp.getSample(m)[1])

validata = ClassificationDataSet(input, target, nb_classes = classes)
for j in xrange(0, validata_temp.getLength()):
    validata.addSample(validata_temp.getSample(j)[0], validata_temp.getSample(j)[1])

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()
validata._convertToOneOfMany()

# TIME TO CREATE A FNN NEURAL NETWORK WITH 2 INPUTS, 3 HIDDEN NEURONS AND 2 OUTPUTS 
FNN_INPUT = 2
FNN_HIDDEN = 15
FNN_OUTPUT = 2
net = buildNetwork(FNN_INPUT, FNN_HIDDEN, FNN_OUTPUT, outclass = SoftmaxLayer, bias=True)
# set up training for back propagation algorithm
# Notice that we set up using the train data
trainer = BackpropTrainer(net, dataset = trndata, momentum = 0.1, verbose = True, weightdecay = 0.01)

# trnerr, valerr = trainer.trainUntilConvergence(dataset=trndata, maxEpochs = 100)
# plt.plot(trnerr, 'b', valerr, 'r')

trainer.trainOnDataset(trndata, 100)

out = net.activateOnDataset(tstdata).argmax(axis=1)

out = net.activateOnDataset(validata).argmax(axis=1)

# print('------------------------------------------------')
pe2 = percentError(out, validata['class'])
print '\n Percent Error :', pe2

print(validata['class'])
print('--------')
print(validata['target'])










