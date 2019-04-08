# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:19:35 2019

@author: rdy11

File 1/5

Description: This file converts the raw CRSP data into an indexed DataFrame.

"""

import pandas as pd
import numpy as np
import progressbar

#import data
filepath = '/Users/Ryan/OneDrive/thesis code/data.csv'
data = pd.DataFrame.from_csv(filepath, index_col=None)

#change data to desired format
def wrangle(data, csv_desired):
    
    #initialize df to be returned at end
    df = pd.DataFrame(data)
    
    #index by date
    df.index = data['DATE']
    
    #remove unnecessary columns
    drop = ['TICKER', 'OPENPRC', 'ASKHI', 'BIDLO', 'ASK', 'BID', 'start',
            'ending', 'RET', 'DATE']
    df.drop(columns=drop, inplace=True)

    #transform df to categorize by security
    def transform(df):
    
        securities = list(set(df['PERMNO']))
        dates = list(set(df.index))
        
        df2 = pd.DataFrame(
                index=dates,
                columns=securities
                )
        
        df2.sort_index(axis=0, inplace=True)
        df2.index.name = 'DATE'
        
        for row in df.itertuples(name=None):
        
            df2.loc[row[0]][row[1]] = abs(row[2])
        
        #export to csv?
        if csv_desired == 1:
            df2.to_csv(r'D:/Python Scripts/thesis/data_out.csv')
        
        else:
            pass
        
        return df2
    
    data_out = {'raw': df, 'cleaned': transform(df)}
    
    return data_out
    
def main(csv_desired):
    
    wrangle(data, csv_desired)