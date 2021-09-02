import pandas as pd
import numpy as np
from random import randrange
from sklearn.model_selection import KFold
from classification import LMKNN

class KFoldCV:
    def __init__(self, dataset, f, lmknn):
        self.__dataset = dataset
        self.__f = f
        self.lmknn = lmknn
        
    @property    
    def dataset(self):
        return self.__dataset

    @property
    def f(self):
        return self.__f

    @dataset.setter
    def dataset(self, input):
        self.__dataset = input

    @f.setter
    def f(self, input):
        self.__f = input
    
    def train_data_dict(self, trainData):
        trainSet = {0:[], 1:[]}
    
        for index in trainData:
                trainSet[index[-1]].append(index[:-1])
        
        return trainSet

    def test_data_dict(self, testData):
        testSet = {0:[], 1:[]}


        for index in testData:
            testSet[index[-1]].append(index[:-1])
            
        return testSet

    def set_accuracy(self, foldAcc, mean):
        accuracyDict = {"Fold accuracy": [], "accuracy": 0}

        for fold in foldAcc:
            accuracyDict["Fold accuracy"].append(fold)

        accuracyDict["accuracy"] = mean

        return accuracyDict

    def execute_cross_val(self):
        results = []
        kf = KFold(n_splits=self.f)
        
        for train_index, test_index in kf.split(self.dataset):
            trainData = self.dataset.iloc[train_index].values.astype(float).tolist()
            testData = self.dataset.iloc[test_index].values.astype(float).tolist()
            train_set = self.train_data_dict(trainData)
            test_set = self.test_data_dict(testData)
            acc = self.lmknn.pred(train_set, test_set, self.lmknn.k)
            results.append(acc)

        accuracy = float(sum(results)/len(results))
        result = self.set_accuracy(results, accuracy)

        return result