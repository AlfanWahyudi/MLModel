import pandas as pd
import numpy as np
from random import randrange
from classification import LMKNN

class KFold:
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

    def cross_val_split(self, dataset, folds):
        dataset_split = []
        df_copy = dataset
        fold_size = int(df_copy.shape[0] / folds)

        for x in range(folds):
            fold = []
            while len(fold) < fold_size:
                random = randrange(df_copy.shape[0])
                index = df_copy.index[random]
                fold.append(df_copy.loc[index].values.tolist())
                df_copy = df_copy.drop(index)

            dataset_split.append(np.asarray(fold))

        return dataset_split

    def set_train_data(self, folds, dataset):
        for fold in folds:
            if fold == folds[0]:
                cv = dataset[fold]
            else:
                cv = np.concatenate((cv, dataset[fold]), axis=0)
        
        return cv

    def fold_list(self, f, number):
        fold = list(range(f))
        fold.pop(number)

        return fold
    
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

    def set_accuracy(self, acc1, acc2):
        accuracyDict = {"fold accuracy": [], "kFold accuracy": 0}

        for fold in acc1:
            accuracyDict["fold accuracy"].append(fold)

        accuracyDict["kFold accuracy"] = acc2

        return accuracyDict

    def execute_cross_val(self):
        data = self.cross_val_split(self.dataset, self.f)
        results = []
        
        for number in range(self.f):
            foldList = self.fold_list(self.f, number)
            cv = self.set_train_data(foldList, data)
            trainList = cv.tolist()
            testList = data[number].tolist()
            train_set = self.train_data_dict(trainList)
            test_set = self.test_data_dict(testList)
            acc = self.lmknn.pred(train_set, test_set, self.lmknn.k)
            results.append(acc)

        kFoldAccuracy = sum(results)/len(results)
        result = self.set_accuracy(results, kFoldAccuracy)

        return result