# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:06:10 2019

@author: rdy11
"""

import numpy as np
import pandas as pd
import math

projectpath = '/Users/Ryan/OneDrive/thesis code/'
filepath = '/Users/Ryan/OneDrive/thesis code/trimmed/data_out_trimmed.csv'
data = pd.read_csv(filepath, na_values=['B', 'C'], index_col=0)
data = pd.DataFrame(data=data, dtype=np.float64)

def MA(a, b, c):
    
    MA_a = data.apply(lambda x: x.rolling(a).mean())
    MA_b = data.apply(lambda x: x.rolling(b).mean())
    MA_c = data.apply(lambda x: x.rolling(c).mean())

    MA = {str(a):MA_a, str(b):MA_b, str(c):MA_c}
    
    return MA
    
def TRB(a, b, c):
    
    def min(n):
        
        min = data.apply(lambda x: x.rolling(n).min())
        
        return min
        
    def max(n):
        
        max = data.apply(lambda x: x.rolling(n).max())
        
        return max
    
    TRB_a = {'min': min(a), 'max': max(a)}
    TRB_b = {'min': min(b), 'max': max(b)}
    TRB_c = {'min': min(c), 'max': max(c)}
    
    TRB = {str(a):TRB_a, str(b):TRB_b, str(c):TRB_c}
    
    return TRB

def ROUND():

    dates = data.index
    securities = data.columns
    round_up = pd.DataFrame(index=data.index, columns=data.columns)
    round_down = pd.DataFrame(index=data.index, columns=data.columns)

    def f_digits(x):
        if np.isnan(x)==False:
            return len(str(int(x)))

    df = pd.DataFrame(index=data.index, columns=data.columns, data=data.applymap(lambda x: f_digits(x)))
    df -= 1
    df = df.applymap(lambda x: min(x, 1))
    digits = pd.DataFrame(index=data.index, columns=data.columns, data=10)
    digits = digits ** df

    round_up = data/digits
    round_up = round_up.applymap(lambda x: np.ceil(x)) * digits
    round_down = data/digits
    round_down = round_down.applymap(lambda x: np.floor(x)) * digits

    ROUND = {'up':round_up, 'down':round_down}

    return ROUND

def ROUND_old():
    
    dates = data.index
    securities = data.columns
    round_up = pd.DataFrame(index=data.index, columns=data.columns)
    round_down = pd.DataFrame(index=data.index, columns=data.columns)

    
    for date in dates: 
        
        for security in securities: 
            
            try: 
                
                round_up.loc[date][security] = 10*(
                        math.ceil(data.loc[date][security]/10))
                
                round_down.loc[date][security] = 10*(
                        math.floor(data.loc[date][security]/10))
                
                print('Passed ' + str(data.loc[date][security]))
                
            except: 
                
                print('Failed ' + str(data.loc[date][security]))
        
    ROUND = {'up':round_up, 'down':round_down}
    
    return ROUND


def main(a, b, c): 
    
    def save(grab, techname):

        A = f'{projectpath}/trimmed/{a}_{techname}'
        B = f'{projectpath}/trimmed/{b}_{techname}'
        C = f'{projectpath}/trimmed/{c}_{techname}'
        
        if techname == 'MA':
            
            grab[str(a)].to_csv(path_or_buf=f'{A}.csv')
            grab[str(b)].to_csv(path_or_buf=f'{B}.csv')
            grab[str(c)].to_csv(path_or_buf=f'{C}.csv')
            
        elif techname == 'TRB':
            
            grab[str(a)]['min'].to_csv(path_or_buf=f'{A}_min.csv')
            grab[str(b)]['min'].to_csv(path_or_buf=f'{B}_min.csv')
            grab[str(c)]['min'].to_csv(path_or_buf=f'{C}_min.csv')
            
            grab[str(a)]['max'].to_csv(path_or_buf=f'{A}_max.csv')
            grab[str(b)]['max'].to_csv(path_or_buf=f'{B}_max.csv')
            grab[str(c)]['max'].to_csv(path_or_buf=f'{C}_max.csv')
            
        if techname == 'ROUND':
            
            grab['up'].to_csv(path_or_buf=f'{projectpath}/trimmed/ROUND_up.csv')
            grab['down'].to_csv(path_or_buf=f'{projectpath}/trimmed/ROUND_down.csv')
        
    save(MA(a, b, c), 'MA')
    save(TRB(a, b, c), 'TRB')
    save(ROUND(), 'ROUND')