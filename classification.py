import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

class LMKNN:
    def __init__(self, k):
        self.__dataTrain = {}
        self.__predict = {}
        self.__k = k

    @property
    def dataTrain(self):
        return self.__dataTrain

    @property
    def predict(self):
        return self.__predict

    @property
    def k(self):
        return self.__k

    @dataTrain.setter
    def dataTrain(self, input):
        self.__dataTrain = input

    @predict.setter
    def predict(self, input):
        self.__predict = input

    @k.setter
    def k(self, input):
        self.__k = input

    def euclidean(self, features, predict):
        return np.linalg.norm(np.array(features)-np.array(predict))

    def calc_distances_data(self, dataTrain, predict):
        distancesDict = {}

        for group in dataTrain:
            distancesList = []
            for index in range(len(dataTrain[group])):
                features = dataTrain[group][index]
                euclideanDistance = self.euclidean(features, predict)
                distancesList.append([euclideanDistance, index])
                distancesDict[group] = distancesList
        
        return distancesDict

    def calc_local_mean(self, distancesData, dataTrain, k):
        local_mean = {}

        for x in distancesData:
            prev = 0
            for y in sorted(distancesData[x])[:k]:
                for label, features in dataTrain.items():
                    if x == label:
                        summation = np.add(np.array(features[y[1]]), prev)
                        prev = summation
                result = np.true_divide(summation, k)
            local_mean[x] = list(result)
            summation = 0
        
        return local_mean

    def calc_closestDist_localMean(self, localMean, predict):
        closestDist = []

        for label in localMean:
            localMeanVector = localMean[label]
            euclideanDistance = self.euclidean(localMeanVector, predict)
            closestDist.append((euclideanDistance, label))

        minimum = np.argmin(closestDist, axis=0)
        result = closestDist[minimum[0]][1]

        return result

    def local_mean_based_knn(self, dataTrain, predict, k):
        distancesData = self.calc_distances_data(dataTrain, predict) 
        localMean = self.calc_local_mean(distancesData, dataTrain, k)
        lmknn = self.calc_closestDist_localMean(localMean, predict)

        return lmknn

    def pred(self, trainSet, testSet, k):
        correct = 0
        total = 0

        for label in testSet:
            for testData in testSet[label]:
                result = self.local_mean_based_knn(trainSet, testData, k)
                if label == result:
                    correct += 1
                total += 1

        return float(correct / total)