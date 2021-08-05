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

    #Preprocessing
    caseFolding = Preprocessing.case_folding(data_choose[column_name[0]])
    st.write("Case Folding")
    st.write(caseFolding)


if __name__ == "__main__":
    main_app()