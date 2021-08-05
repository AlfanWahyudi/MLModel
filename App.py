from os import remove
from re import escape
from altair.vegalite.v4.schema.core import Transform
import streamlit as st
import numpy as np
import pandas as pd
import time
import Data
import Preprocessing
    
def main_app():
    task = st.sidebar.radio('Navigasi', ['Data Cleaning', 'TF-IDF', 'LMKNN'], 0)

    # Add a selectbox to the sidebar:
    add_selectbox = st.sidebar.selectbox(
        'How would you like to be contacted?',
        ('Email', 'Home phone', 'Mobile phone')
    )

    # Add a slider to the sidebar:
    add_slider = st.sidebar.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0)
    )

    csv_file = Data.input_csv()
    if not csv_file:
        st.write("Upload CSV file to get started")
        return
    
    df = pd.read_csv(csv_file)   
    Data.explore(df)
    data_choose = Data.choose_column(df)
    st.write(data_choose)
    
    #get column name of data 
    column_name = data_choose.columns.values.tolist()

    # dataset = data_choose[column_name[0]].str.encode('ascii', 'ignore')

    #Preprocessing
    st.write("Case Folding")
    caseFolding = Preprocessing.case_folding(data_choose[column_name[0]])
    st.write(caseFolding)

    st.write("Convert Emoticon")
    convertEmot =  caseFolding.apply(Preprocessing.convert_emot)
    st.write(convertEmot)

    st.write("Convert Slang Word")
    slangWord = convertEmot.apply(Preprocessing.convert_slang)
    st.write(slangWord)

    st.write("Cleaning")
    cleanHtml =   slangWord.apply(Preprocessing.clean_html)
    removeTxtSpecial =  cleanHtml.apply(Preprocessing.remove_text_special)
    removeWhitespace =  removeTxtSpecial.apply(Preprocessing.remove_whitespace)    
    removeSingleChar = removeWhitespace.apply(Preprocessing.remove_single_char)
    removeNonASCII = removeSingleChar.apply(Preprocessing.remove_non_ASCII)
    removePuntuation = removeNonASCII.apply(Preprocessing.remove_punctuation)
    removeNumber = removePuntuation.apply(Preprocessing.remove_number)
    st.write(removeNumber)


    st.write("Convert Negation")
    convertNegation = removeNumber.apply(Preprocessing.convert_slang)
    st.write(convertNegation)

    st.write("Tokenization")
    tokenization = convertNegation.apply(Preprocessing.tokenization)
    st.write(tokenization)

    st.write("Stoword Removal")
    stopwordsRemoval = tokenization.apply(Preprocessing.stopwords_removal)
    st.write(stopwordsRemoval)

    
if __name__ == "__main__":
    main_app()