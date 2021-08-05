import pandas as pd
import numpy as np
import string 
import re #regex library
import itertools
from nltk.corpus import stopwords

def case_folding(text):
    return text.str.lower()

def clean_html(text):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', text)

    return cleantext

def remove_text_special(text):
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    text = ' '.join(re.sub(r"([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())

    return text.replace("http://", " ").replace("https://", " ")

def remove_whitespace(text):
    return " ".join(text.split())

def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

def remove_non_ASCII(text):
    return text.encode('ascii', 'replace').decode('ascii')

def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

def remove_number(text):
    return ''.join((word for word in text if not word.isdigit()))

def many_letters(word):
    word = list(word)
    Letters = []
    
    for letter in word:
        if letter not in Letters:
            Letters.append(letter)
            
    return len(Letters)

def remove_duplicate(text):
    words = text.split()
    
    for index, word in enumerate(words):
        numLetters = many_letters(word)
        if numLetters > 2:
            words[index] = ''.join(c[0] for c in itertools.groupby(word))
        else:
            words[index] = ''.join(sorted(set(word), key=word.index))     

    return ' '.join(words)

def move_list(data_dictionary):
    newList = []
    lenght = len(data_dictionary)
    
    for row in range(lenght):
        txt_split = re.split(r"[\t:]+", data_dictionary[row])
        newList.append(txt_split)
        
    return newList

def join_two_dict(dict1, dict2):
    dictSlangWord = pd.read_csv(dict1, names= ["dict1"], header = None)
    dictKBBA = pd.read_csv(dict2, names= ["dict2"], header = None)

    slangWord = move_list(dictSlangWord['dict1'])
    KBBA = move_list(dictKBBA['dict2'])

    for x in KBBA:
        slangWord.append(x)
    
    return slangWord

def convert_slang(text):
    dictionary =  join_two_dict('slangword.txt','kbba.txt')
    removeDuplicate = remove_duplicate(text)
    words = removeDuplicate.split()

    for word in range(len(words)):
        for index in range(len(dictionary)):
            if words[word] == dictionary[index][0]:
                words[word] = dictionary[index][1]
                
    return ' '.join(words)

emoticon = {"senang" : [">:]", ":-)", ":)", ":o)", ":]", ":3", ":c)", ":>", "=]", "8)", "=)", ":}", ":^)"], 
            "tertawa" : [">:D", ":-D", ":D", "8-D", "8D", "x-D", "xD", "XD", "=-D", "=D", "=-3", "=3"], 
            "sedih" : [">:[", ":-(", ":(", ":-c", ":c", ":-<", ":<", ":-[", ":[", ":{", ">", ".><.", "<>", ".<", ":’("],
            "jilat" : [">:P", ":-P", ":P", "X-P", "x-p", "xp", "XP", ":-p", ":p", "=p", ":-Þ", ":Þ", ":-b", ":b"],
            "terkejut" : [">:o", ">:O", ":-O", ":O", "°o°", "°O°", ":O", "o_O", "o.O", "8-0"],
            "kesal" : [">:/", ":-/", ":-.", ":/", "=/"],
            "ekspresi datar" : [":|", ":-|"]}

def convert_emot(text):
    words = text.split()
    
    for word in range(len(words)):
        for txt, emots in emoticon.items():
            for emot in range(len(emots)):
                if words[word] == emots[emot]:
                    words[word] = txt
                    
    return ' '.join(words)

negationWord = ['ga', 'ngga', 'tidak', 
               'bkn', 'tida', 'tak', 
               'jangan', 'enggan', 'gak']

def convert_negation(text):
    words = text.split()
    
    for index, word in enumerate(words):
        for negation in range(len(negationWord)):
            if words[index] == negationWord[negation]:
                nxt = index + 1 
                words[index] = negationWord[negation] + words[nxt]
                words.pop(nxt)
            
    return ' '.join(words)

def tokenization(text):
    token = re.split('\W+', text)
    return token

#stopword
def combine_stopword_dict():
    list_stopwords = stopwords.words('indonesian')
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                        'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                        'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                        'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                        'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                        '&amp', 'yah'])

    txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)
    list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
    set_stopwords = set(list_stopwords)

    return set_stopwords

def stopwords_removal(words):
    stopwordsDict = combine_stopword_dict()
    return [word for word in words if word not in stopwordsDict]


if __name__ == "__main__":
    case_folding
    clean_html
    remove_text_special
    remove_duplicate
    remove_non_ASCII
    remove_number
    remove_punctuation
    remove_single_char
    remove_text_special
    remove_whitespace
    join_two_dict
    many_letters
    convert_slang
    convert_emot
    convert_negation
    tokenization