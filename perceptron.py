# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:20:00 2019

@author: Dr AQ
"""
import sys
import numpy as np
import pandas as pd
import random
import csv


# Importing the dataset
dataset = pd.read_csv('data.csv')
#dataset.insert(0, 'BIAS', 1)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train ,X_test ,y_train ,y_test=train_test_split(X ,y ,test_size=0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X.fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)

X_train = np.insert(X_train,0,[1]*X_train.shape[0],axis=1)
X_test = np.insert(X_test,0,[1]*X_test.shape[0],axis=1)

class Perceptron():

    # The constructor of our class.
    def __init__(self):
        self.X = X_train
        self.desiredOutput = y_train
        self.weights = [random.uniform(-0.5, 0.5)
                        for i in range(0, len(X_train[0]))]
        f = open('Results.txt', 'w')
        f.write('AQ PERCPETRON RESULTS')
        f.close()

    def predict(self, inputs):
        threshold = 0.0
        total_activation = 0.0
        for input, weight in zip(inputs, self.weights):
            total_activation += input*weight
        return 1.0 if total_activation >= threshold else 0.0

    def updateWeights(self, error, inp, alpha=0.1): 
        for j in range(len(self.weights)):
            self.weights[j] = self.weights[j]+(alpha*error*inp[j])
    

    def recall(self, tp, fn):
        result = tp / (tp + fn)
        return result * 100

    def precision(self, tp, fp):
        result = tp / (tp + fp)
        return result * 100
        
    def accuracy(self, tp, tn, fp, fn):
        s = tp + tn
        s2 = tp + tn + fp + fn
        result = s / s2
        return result * 100

    def learn(self):
        totalError = 1
        epoch = 1
        with open('weights.csv', 'w', newline='') as file:

            while totalError != 0 and epoch != 10:
                totalError = 0
                for i in range(0, len(self.desiredOutput)):
                    predicted = self.predict(self.X[i])
                    error = self.desiredOutput[i] - predicted
                    if error != 0:
                        self.updateWeights(error, self.X[i])
                        totalError = totalError + 1
                    print("##########################################")
                    print("\nEpoch %d:\n " % epoch)
                    print("\nPrediction:\n ", predicted)
                    print("\nError:\n ", error)
                    print("\nFinal Weights:\n ", self.weights)
                    print("\n##########################################")
                    f = open('Results.txt', 'a')
                    f.write('\n')
                    f.write(
                        "###########################################################################")
                    f.write('\n')
                    f.write('Epoch :')
                    f.write('\n')
                    f.write(str(epoch))
                    f.write('\n')
                    f.write('Prediction:')
                    f.write('\n')
                    f.write(str(predicted))
                    f.write('\n')
                    f.write('Error:')
                    f.write(str(error))
                    f.write('\n')
                    f.write('Final Weights:')
                    f.write('\n')
                    f.write(str(self.weights))
                    f.write('\n')
                    f.write(
                        "###########################################################################")
                    f.write('\n')
                    f.close()    
                epoch = epoch + 1
            writer = csv.writer(file)
            writer.writerow(self.weights)

    def testModel(self):
        weight = pd.read_csv('weights.csv')
        self.weights = list(map(float, weight))
        self.X = X_test
        self.desiredOutput = y_test
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(0, len(self.desiredOutput)):
            predicted = self.predict(self.X[i])
            if predicted == self.desiredOutput[i]:
                if predicted == 1:
                    tp = tp + 1
                else:
                    tn = tn + 1
            else:
                if predicted == 0:
                    fn = fn + 1
                else:
                    fp = fp + 1

        print('The AQ Model is ' + str(self.accuracy(tp, tn, fp, fn)) + '% Accurate')
        print('The AQ Model Precision is ' + str(self.precision(tp, fp)))
        print('The AQ Model Recall is ' + str(self.recall(tp, fn)))


###################################################################
ppn = Perceptron()


def check(proc):
    if proc == 'train':
        ppn.learn()
    else:
        ppn.testModel()


check(sys.argv[1])

#ppn.learn()
#ppn.testModel()