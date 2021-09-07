import pandas as pd
import numpy as np
import string 
import re 
import itertools
from nltk.corpus import stopwords
from gensim.utils import tokenize

class Preprocessing:
    def __init__(self, data):
        self.__data = data

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, input):
        self.__data = input
class CaseFold(Preprocessing):
    def __init__(self, data):
        super().__init__(data)

    def case_folding(self):
        return self.data.str.lower()
    
class Noise(Preprocessing):
    def __init__(self, data):
        super().__init__(data)
    
    def clean_html(self, data):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, ' ', data)

        return cleantext

    def remove_text_special(self, data):
        data = data.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
        data = ' '.join(re.sub(r"([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", data).split())

        return data.replace("http://", " ").replace("https://", " ")

    def remove_punctuation(self, data):
        return data.translate(str.maketrans("","",string.punctuation))

    def remove_number(self, data):
        return ''.join((word for word in data if not word.isdigit()))

    def remove_whitespace(self, data):
        return " ".join(data.split())

    def remove_non_ASCII(self, data):
        return data.encode('ascii', 'replace').decode('ascii')

    def remove_single_char(self, data):
        return re.sub(r"\b[a-zA-Z]\b", "", data)

class Normalization(Preprocessing):
    def __init__(self, data):
        super().__init__(data)
        self.emoticon_dict = {"senang" : [">:]", ":-)", ":)", ":o)", ":]", ":3", ":c)", ":>", "=]", "8)", "=)", ":}", ":^)"], 
            "tertawa" : [">:d", ":-d", ":d", "8-d", "8d", "x-d", "xd", "=-d", "=d", "=-3", "=3"], 
            "sedih" : [">:[", ":-(", ":(", ":-c", ":c", ":-<", ":<", ":-[", ":[", ":{", "> ", ".><.", "<>", ".<", ":’("],
            "jilat" : [">:p", "x-p", "xp", ":-p", ":p", "=p", ":-Þ", ":Þ", ":-b", ":b"],
            "terkejut" : [">:o", ">:O", ":-O", ":O", "°o°", "°O°", ":O", "o_O", "o.O", "8-0"],
            "kesal" : [">:/", ":-/", ":-.", ":/", "=/"],
            "ekspresi datar" : [":|", ":-|"]}

    def many_letters(self, word):
        word = list(word)
        Letters = []
        
        for letter in word:
            if letter not in Letters:
                Letters.append(letter)
                
        return len(Letters)

    def remove_duplicate(self, data):
        words = data.split()
        
        for index, word in enumerate(words):
            numLetters = self.many_letters(word)
            if numLetters == 2 and len(word) == 3:
                words[index] = ''.join(c[0] for c in itertools.groupby(word))
            elif numLetters > 2:
                words[index] = ''.join(c[0] for c in itertools.groupby(word))
            elif numLetters <= 2:
                words[index] = ''.join(sorted(set(word), key=word.index))

        return ' '.join(words)

    def move_list(self, data_dictionary):
        newList = []
        lenght = len(data_dictionary)
        
        for row in range(lenght):
            txt_split = re.split(r"[\t:]+", data_dictionary[row])
            newList.append(txt_split)
            
        return newList

    def join_two_dict(self, dict1, dict2):
        dictionary1 = pd.read_csv(dict1, names= ["dict1"], header = None)
        dictionary2 = pd.read_csv(dict2, names= ["dict2"], header = None)

        dict1_list = self.move_list(dictionary1['dict1'])
        dict2_list = self.move_list(dictionary2['dict2'])

        for word in dict2_list:
            dict1_list.append(word)
        
        return dict1_list

    def convert_slang(self, data):
        dictionary =  self.join_two_dict('data/slangword.txt','data/kbba.txt')
        removeDuplicate = self.remove_duplicate(data)
        words = removeDuplicate.split()

        for word in range(len(words)):
            for index in range(len(dictionary)):
                if words[word] == dictionary[index][0]:
                    words[word] = dictionary[index][1]
                    
        return ' '.join(words)

    def convert_emot(self, data):
        words = data.split()
        
        for word in range(len(words)):
            for txt, emots in self.emoticon_dict.items():
                for emot in range(len(emots)):
                    result = words[word].replace(emots[emot], " "+txt+" ")
                    words[word] = result
                        
        return ' '.join(words)

class Stopword(Preprocessing):
    def __init__(self, data):
        super().__init__(data)

    def combine_stopword_dict(self):
        list_stopwords = stopwords.words('indonesian')
        list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                            'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                            'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                            'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                            'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                            'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                            '&amp', 'yah'])

        txt_stopword = pd.read_csv("data/stopwords.txt", names= ["stopwords"], header = None)
        list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
        set_stopwords = set(list_stopwords)

        return set_stopwords

    def stopwords_removal(self, data):
        words = data.split()
        stopwordsDict = self.combine_stopword_dict()
        
        return ' '.join(word for word in words if word not in stopwordsDict)

class Negation(Preprocessing):
    def __init__(self, data):
        super().__init__(data)
        self.negationWord = ['ga', 'ngga', 'tidak', 
                'bkn', 'tida', 'tak', 
                'jangan', 'enggan', 'gak']
    
    def convert_negation(self, data):
        words = data.split()
        
        for index, word in enumerate(words):
            for negation in range(len(self.negationWord)):
                if words[index] == self.negationWord[negation]:
                    nxt = index + 1 
                    if nxt != len(words):
                        words[index] = self.negationWord[negation] + words[nxt]
                        words.pop(nxt)
                
        return ' '.join(words)

class Tokenization(Preprocessing):
    def __init__(self, data):
        super().__init__(data)
    
    def token(self, data):
        token = list(tokenize(data))
        
        return token

