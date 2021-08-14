import streamlit as st
import numpy as np
import pandas as pd
from streamlit.elements.number_input import Number
from data import *
from preprocessing import *
from TfIdf import *
from classification import *
from evaluate import *
    
class App:
    def __init__(self):
        pass

    def preprocessing(self, text_data):
        st.write("Case Folding")
        caseFold = CaseFold(text_data)
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

        st.write("Stopword Removal")
        stopword = Stopword(removeSingleChar)
        stopwordRemoval = removeSingleChar.apply(stopword.stopwords_removal)
        st.write(stopwordRemoval)

        st.write("Convert Negation")
        negation = Negation(stopwordRemoval)
        convertNegation = stopwordRemoval.apply(negation.convert_negation)
        st.write(convertNegation)
        
        st.write("Tokenization")
        tokenization = Tokenization(convertNegation)
        token = convertNegation.apply(tokenization.token)
        st.write(token)

        return token

    def main_app(self):
        dataComment = Data()
        csv_file = dataComment.input_csv()

        if not csv_file:
            st.info("Upload CSV file to get started")
            return

        df = pd.read_csv(csv_file)
        data_choose = dataComment.choose_column(df)
        
        st.subheader("Dataset")
        st.write(data_choose)
        column_name = data_choose.columns.values.tolist()
        
        st.subheader("Preprocessing")
        cleanData = self.preprocessing(data_choose[column_name[0]])
 
        st.subheader("TF-IDF")
        tfIdf = TfIdf(cleanData)
        tf = cleanData.apply(tfIdf.calc_tf)
        df = tfIdf.calc_df(cleanData)
        idf = tfIdf.calc_idf(len(cleanData), df)
        tfIdfResult = tfIdf.tf_idf(tf, idf)

        dfResult = pd.DataFrame(tfIdfResult)
        addLabel = dfResult.assign(Label = data_choose[column_name[1]])
        dataset = addLabel.fillna(0)
        dataset.to_csv("hasil ekstraksi fitur.csv")
        st.write(dataset)

        st.subheader("Local Mean-Based k-Nearest Neighbor")
        with st.form(key='lmknn'):
            k = st.text_input(label='Input K')
            f = st.text_input(label="Input Fold")
            btnSubmit = st.form_submit_button(label='Predict')

        if btnSubmit:
            # dataset = pd.DataFrame(final)
            lmknn = LMKNN(int(k))
            kfold = KFold(dataset, int(f), lmknn)
            result = kfold.execute_cross_val()

            for key in result:
                if key == "Fold accuracy":
                    st.write(key)
                    number = 0
                    for value in result[key]:
                        number += 1
                        st.write(str(number),")", str(value))
                else:
                    st.write("Accuracy: " , str(result[key]))

if __name__ == "__main__":
    app = App()
    app.main_app()