# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:57:17 2019

@author: rdy11
"""

import pandas as pd
import numpy as np

projectpath = '/Users/Ryan/OneDrive/thesis/trimmed'
csvpath = '/Users/Ryan/OneDrive/thesis/trimmed'
filepath = '/Users/Ryan/OneDrive/thesis/returns_out_trimmed.csv'
returns = pd.DataFrame.from_csv(filepath)

def import_signals():

    indicators = ['MA_a', 'MA_b', 'MA_c', 
                'TRB_a_min', 'TRB_b_min', 'TRB_c_min', 
                'TRB_a_max', 'TRB_b_max', 'TRB_c_max', 
                'ROUND_up', 'ROUND_down']
    
    csvs = [pd.DataFrame.from_csv(f'{csvpath}/MA_a_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/MA_b_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/MA_c_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/TRB_a_min_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/TRB_b_min_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/TRB_c_min_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/TRB_a_max_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/TRB_b_max_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/TRB_c_max_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/ROUND_up_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/ROUND_down_out.csv')]
    
    signals_dict = dict(zip(indicators, csvs))

    return signals_dict
    
def generate_container():
    
    securities = returns.columns
    
    indicators = ['MA_a', 'MA_b', 'MA_c', 
                'TRB_a_min', 'TRB_b_min', 'TRB_c_min', 
                'TRB_a_max', 'TRB_b_max', 'TRB_c_max', 
                'ROUND_up', 'ROUND_down']
    
    container = {indicator: {security: 
        {'all': [], 'signal': []}
        for security in securities}
        for indicator in indicators
            }
    
    return container

def fill_container(container, signals_dict):
    
    securities = returns.columns
    
    indicators = ['MA_a', 'MA_b', 'MA_c', 
                'TRB_a_min', 'TRB_b_min', 'TRB_c_min', 
                'TRB_a_max', 'TRB_b_max', 'TRB_c_max', 
                'ROUND_up', 'ROUND_down']
    
    for indicator in indicators: 
        
        for security in securities: 
        
            container[indicator][security]['all'] = returns[security].dropna()
        
            container[indicator][security]['signal'] = returns[security].mask(
                signals_dict[indicator][security]==True)
    
    return container