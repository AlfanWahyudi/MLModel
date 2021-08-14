from nltk.corpus.reader.chasen import test
import pandas as pd
import numpy as np
from random import randrange
thislist = [[0,1,1,0,1],
            [0,0,1,1,1], 
			[0,1,0,0,0], 
            [1,1,1,0,1], 
            [1,0,0,0,1], 
            [1,1,0,1,0],
            [0,0,1,1,1], 
            [0,1,0,0,0], 
            [0,0,1,0,0], 
            [1,0,1,0,0]]

def cross_val_split(dataset, folds):
    dataset_split = []
    df_copy = dataset

    fold_size = int(df_copy.shape[0] / folds)

    for x in range(folds):
        fold = []
        
        while len(fold) < fold_size:
            random = randrange(df_copy.shape[0])

            index = df_copy.index[random]
            print("index ",index)

            fold.append(df_copy.loc[index].values.tolist())

            df_copy = df_copy.drop(index)
        print("")

        dataset_split.append(np.asarray(fold))

    return dataset_split

def euclidean(features, predict):
    return np.linalg.norm(np.array(features)-np.array(predict))

def calc_distances_data(data, predict):
    distancesDict = {}

    for group in data:
        distancesList = []
        for index in range(len(data[group])):
            features = data[group][index]
            euclideanDistance = euclidean(features, predict)
            distancesList.append([euclideanDistance, index])
            distancesDict[group] = distancesList
    
    return distancesDict

def calc_local_mean(distances_data, data, k):
    local_mean = {}

    for x in distances_data:
        prev = 0
        for y in sorted(distances_data[x])[:k]:
            for label, features in data.items():
                if x == label:
                    summation = np.add(np.array(features[y[1]]), prev)
                    prev = summation
            result = np.true_divide(summation, len(sorted(distances_data[x])[:k]))
        local_mean[x] = list(result)
        summation = 0
    
    return local_mean

def calc_closestDist_localMean(localMean, predict):
    closestDist = []

    for label in localMean:
        localMeanVector = localMean[label]
        euclideanDistance = euclidean(localMeanVector, predict)
        closestDist.append((euclideanDistance, label))

    minimum = np.argmin(closestDist, axis=0)
    result = closestDist[minimum[0]][1]

    return result

def local_mean_based_knn(data, predict, k=3):
    distancesData = calc_distances_data(data, predict)    
    localMean = calc_local_mean(distancesData, data, k)
    lmknn = calc_closestDist_localMean(localMean, predict)

    return lmknn

def predict(trainSet, testSet, k):
    benar = 0
    total = 0

    #melakukan klasifikasi lmknn
    for label in testSet:
        for testData in testSet[label]:
            hasil = local_mean_based_knn(trainSet, testData, k)
            print("hasil: ",hasil)
            if label == hasil:
                benar += 1
            total += 1

    return float(benar / total)


def set_train_data(folds, data):
    for fold in folds:
            # print(r[0])
        if fold == folds[0]:
            cv = data[fold]
                # print(cv)
        else:
                # print("ELSE :",cv)
            cv = np.concatenate((cv, data[fold]), axis=0)
            # print("\ncross validation",cv)
            # print(j)
    
    return cv

def fold_list(f, number):
    fold = list(range(f))
    fold.pop(number)

    return fold

def train_data_dict(trainData):
    trainSet = {0:[], 1:[]}

    for index in trainData:
        trainSet[index[-1]].append(index[:-1])
    
    return trainSet


def test_data_dict(testData):
    testSet = {0:[], 1:[]}

    for index in testData:
        testSet[index[-1]].append(index[:-1])
    
    return testSet

def set_accuracy(acc1, acc2):
    accuracyDict = {"fold accuracy": [], "kFold accuracy": 0}

    for x in acc1:
        accuracyDict["fold accuracy"].append(x)

    accuracyDict["kFold accuracy"] = acc2

    return accuracyDict


def kFoldCV(dataset, f=5, k=3):
    data = cross_val_split(dataset, f)
    results = []
    
    for number in range(f):
        foldList = fold_list(f, number)

        #menentukan data latih
        cv = set_train_data(foldList, data)

        # for j in fold:
            
        #     # print(r[0])
        #     if j == fold[0]:
        #         cv = data[j]
        #         # print(cv)
        #     else:
        #         # print("ELSE :",cv)
        #         cv = np.concatenate((cv, data[j]), axis=0)
        #     # print("\ncross validation",cv)
        #     # print(j)
        
        # print("data train", cv)

        #merubah numpy ke list untuk data latih dan data uji
        trainList = cv.tolist()
        testList = data[number].tolist()
        # print("test data = ", data[i])
        # print("==========================================")

        #merubah bentuk list ke dictinary data latih dan uji
        train_set = train_data_dict(trainList)
        print("data latih : ", train_set)
        test_set = test_data_dict(testList)
        print("\ndata uji", test_set)
        # for index in trainData:
        #     train_set[index[-1]].append(index[:-1])

        # for index in testData:
        #     test_set[index[-1]].append(index[:-1])

        # benar = 0
        # total = 0

        #melakukan klasifikasi lmknn
        acc = predict(train_set, test_set, k)
        results.append(acc)

        # for label in test_set:
        #     for testData in test_set[label]:
        #         hasil = local_mean_based_knn(train_set, testData, k)
        #         print("hasil: ",hasil)
        #         if label == hasil:
        #             benar += 1
        #         total += 1
        #     print("benar: ",benar)
        # results.append(float(benar/total))

        # print("\ndata latih",train_set)
        # train_set.clear()
        # print(train_set)
        # test_set.clear()

    kFoldAccuracy = (sum(results)/len(results)) * 100
    result = set_accuracy(results, kFoldAccuracy)

    return result

csv_file = pd.read_csv('cobaLMKNN.csv')

df = pd.DataFrame(thislist)

cross_validation = kFoldCV(df, f=5, k=3)
print(cross_validation)

# sum of list cv resulst
# print(sum(cross_validation))

# accuracy of lmknn using kfold
# kFoldAccuracy = sum(cross_validation)/len(cross_validation)
# print(kFoldAccuracy)
# print(float(sum(cross_validation)/len(cross_validation)))