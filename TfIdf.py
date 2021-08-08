import pandas as pd
import numpy as np
import ast
import math

class TfIdf:
    
    def __init__(self, dataClean):
        self.__dataClean = dataClean

    @property
    def dataClean(self):
        return self.__dataClean

    @dataClean.setter
    def dataClean(self, input):
        self.__dataClean = input


    def convert_text_list(self, dataClean):
        texts = ast.literal_eval(dataClean)
        
        return [text for text in texts]

    def calc_tf(self, dataClean):
        TF_dict = {}

        for elem in dataClean:
            if elem in TF_dict:
                TF_dict[elem] += 1
            else:
                TF_dict[elem] = 1

        for term in TF_dict:
            TF_dict[term] = TF_dict[term] / len(dataClean)

        return TF_dict
        

    def calc_df(self, dataClean):
        count_DF = {}

        for word in dataClean:
            for term in word:
                if term in count_DF:
                    count_DF[term] += 1
                else:
                    count_DF[term] = 1

        return count_DF

    def calc_idf(self, n_word, df):
        IDF_dict = {}
        
        for term in df:
            IDF_dict[term] = math.log10(n_word / df[term])

        return IDF_dict

    def tf_idf(self, tf, idf):
        TF_IDF_list = []

        for numb, document in tf.items():
            TF_IDF_dict = {word: val * idf[word] for word, val in document.items()}     
            TF_IDF_list.append(TF_IDF_dict)

        return TF_IDF_list

    def label_to_TfIdf(self):
        pass

    
    