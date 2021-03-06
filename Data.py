import streamlit as st
import pandas as pd
import numpy as np

class Data:
    def __init__(self):
        self.df = None
        self.dataset = None

    @property
    def df(self):
        return self.__df
        
    @property
    def dataset(self):
        return self.__dataset
    
    @df.setter
    def df(self, input):
        self.__df = input

    @dataset.setter
    def dataset(self, input):
        self.__dataset = input

    def input_csv(self):
        self.dataset = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

        return self.dataset

    def explore(self, df):
        # DATA
        st.write('Data:')
        st.write(df)
        # SUMMARY
        df_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
        numerical_cols = df_types[~df_types['Data Type'].isin(['object',
                        'bool'])].index.values
        df_types['Count'] = df.count()
        df_types['Unique Values'] = df.nunique()
        df_types['Min'] = df[numerical_cols].min()
        df_types['Max'] = df[numerical_cols].max()
        df_types['Average'] = df[numerical_cols].mean()
        df_types['Median'] = df[numerical_cols].median()
        df_types['St. Dev.'] = df[numerical_cols].std()
        st.write('Summary:')
        st.write(df_types)


    def choose_column(self, df):
        cols = st.sidebar.multiselect('Choose Columns', 
                                   df.columns.tolist(),
                                   df.columns.tolist())
        df = df[cols]

        return df