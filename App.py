import streamlit as st
import numpy as np
import pandas as pd
import Data
from Preprocessing import CaseFold, Noise, Normalization, Tokenization, Stopword, Negation
    
class App:
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
        # caseFolding = Preprocessing.case_folding(data_choose[column_name[0]])
    
  
        # caseFolding
        caseFold = CaseFold(data_choose[column_name[0]])
        caseFold = caseFold.case_folding()
        st.write(caseFold)

        st.write("Normalization")
        normalization = Normalization(caseFold)
        convertEmot =  caseFold.apply(normalization.convert_emot)
        removeDuplicate = convertEmot.apply(normalization.remove_duplicate)
        slangWord = removeDuplicate.apply(normalization.convert_slang)
        st.write(slangWord)

        st.write("Noise Removal")
        noiseRemoval = Noise(slangWord)
        cleanHtml = slangWord.apply(noiseRemoval.clean_html)
        removeSpecial = cleanHtml.apply(noiseRemoval.remove_text_special)
        removeNonASCII = removeSpecial.apply(noiseRemoval.remove_non_ASCII)
        removePuntuation = removeNonASCII.apply(noiseRemoval.remove_punctuation)
        removeNumber =  removePuntuation.apply(noiseRemoval.remove_number)
        removeWhitespace = removeNumber.apply(noiseRemoval.remove_whitespace)
        removeSingleChar = removeWhitespace.apply(noiseRemoval.remove_single_char)
        st.write(removeSingleChar)

        st.write("Convert Negation")
        negasi = Negation(removeSingleChar)
        convertNegation = removeSingleChar.apply(negasi.convert_negation)
        st.write(convertNegation)
        
        st.write("Tokenization")
        tokenization = Tokenization(convertNegation)
        token = convertNegation.apply(tokenization.token)
        st.write(token)

        st.write("Stopword Removal")
        stopword = Stopword(token)
        stopwordRemoval = token.apply(stopword.stopwords_removal)
        st.write(stopwordRemoval)

        
    if __name__ == "__main__":
        main_app()