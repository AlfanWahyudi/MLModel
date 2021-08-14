class KFold:
    def __init__(self, dataset, f, lmknn):
        self.dataset = dataset
        self.train_set = "this is train data"
        self.test_set = "this is test data"
        self.lmknn = lmknn
        self.f = f


    def execute_cross_val(self):
        print("dataset : ", self.dataset)
        print("folds : ", self.f)
        self.lmknn.predict(self.train_set, self.test_set, lmknn.k)

class LMKNN:
    def __init__(self, k):
        self.trainSet = None
        self.testSet = None
        self.k = k

    def predict(self, trainset, testSet, k):
        print("\ndata latih : ", trainset)
        print("data uji : ", testSet)
        print("akurasi lmknn : 80% " , "dengan K sebesar ", k)


if __name__ == "__main__" :
    lmknn = LMKNN(6)
    kFold = KFold('dataset', 5, lmknn)
    kFold.execute_cross_val()


    
    