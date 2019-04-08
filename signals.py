# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:28:49 2019

@author: rdy11
"""

import numpy as np
import pandas as pd

projectpath = '/Users/Ryan/OneDrive/thesis'
csvpath = '/Users/Ryan/OneDrive/thesis code/trimmed'
filepath = '/Users/Ryan/OneDrive/thesis code/trimmed/data_out_trimmed.csv'
data = pd.DataFrame.from_csv(filepath)
data = pd.DataFrame(data=data, dtype=np.float64)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

BEGIN MA

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def generate_MA(a, b, c, j):
    
    MA_a_csv = pd.DataFrame.from_csv(f'{csvpath}/{a}_MA.csv')
    MA_b_csv = pd.DataFrame.from_csv(f'{csvpath}/{b}_MA.csv')
    MA_c_csv = pd.DataFrame.from_csv(f'{csvpath}/{c}_MA.csv')
    
    def gen_a():
        
        logs = np.log(MA_a_csv / data)
        ans = ((j > logs) & (logs > -j)).replace({True: 1, False: 0})
        ans.to_csv(path_or_buf=f'{csvpath}/MA_a_out.csv')
                    
    def gen_b(): 
        
        logs = np.log(MA_b_csv / data)
        ans = ((j > logs) & (logs > -j)).replace({True: 1, False: 0})
        ans.to_csv(path_or_buf=f'{csvpath}/MA_b_out.csv')
    
    def gen_c():
        
        logs = np.log(MA_c_csv / data)
        ans = ((j > logs) & (logs > -j)).replace({True: 1, False: 0})
        ans.to_csv(path_or_buf=f'{csvpath}/MA_c_out.csv')
        
    gen_a()
    gen_b()
    gen_c()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

BEGIN TRB

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def generate_TRB(a, b, c, j):
    
    TRB_a_min_csv = pd.DataFrame.from_csv(f'{csvpath}/{a}_TRB_min.csv')
    TRB_b_min_csv = pd.DataFrame.from_csv(f'{csvpath}/{b}_TRB_min.csv')
    TRB_c_min_csv = pd.DataFrame.from_csv(f'{csvpath}/{c}_TRB_min.csv')
    TRB_a_max_csv = pd.DataFrame.from_csv(f'{csvpath}/{a}_TRB_max.csv')
    TRB_b_max_csv = pd.DataFrame.from_csv(f'{csvpath}/{b}_TRB_max.csv')
    TRB_c_max_csv = pd.DataFrame.from_csv(f'{csvpath}/{c}_TRB_max.csv')
    
    def gen_a_min():
        
        logs = np.log(TRB_a_min_csv / data)
        ans = ((j > logs) & (logs > -j)).replace({True: 1, False: 0})
        ans.to_csv(path_or_buf=f'{csvpath}/TRB_a_min_out.csv')
                    
    def gen_b_min(): 
        
        logs = np.log(TRB_b_min_csv / data)
        ans = ((j > logs) & (logs > -j)).replace({True: 1, False: 0})
        ans.to_csv(path_or_buf=f'{csvpath}/TRB_b_min_out.csv')
    
    def gen_c_min():

        logs = np.log(TRB_c_min_csv / data)
        ans = ((j > logs) & (logs > -j)).replace({True: 1, False: 0})
        ans.to_csv(path_or_buf=f'{csvpath}/TRB_c_min_out.csv')
        
    def gen_a_max():
        
        logs = np.log(TRB_a_max_csv / data)
        ans = ((j > logs) & (logs > -j)).replace({True: 1, False: 0})
        ans.to_csv(path_or_buf=f'{csvpath}/TRB_a_max_out.csv')
                    
    def gen_b_max(): 
        
        logs = np.log(TRB_b_max_csv / data)
        ans = ((j > logs) & (logs > -j)).replace({True: 1, False: 0})
        ans.to_csv(path_or_buf=f'{csvpath}/TRB_b_max_out.csv')
    
    def gen_c_max():
        
        logs = np.log(TRB_c_max_csv / data)
        ans = ((j > logs) & (logs > -j)).replace({True: 1, False: 0})
        ans.to_csv(path_or_buf=f'{csvpath}/TRB_c_max_out.csv')
        
    gen_a_min()
    gen_b_min()
    gen_c_min()
    gen_a_max()
    gen_b_max()
    gen_c_max()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

BEGIN ROUND

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def generate_ROUND(j):

    ROUND_up_csv = pd.DataFrame.from_csv(f'{csvpath}/ROUND_up.csv')
    ROUND_down_csv = pd.DataFrame.from_csv(f'{csvpath}/ROUND_down.csv')

    def gen_up():
        
        logs = np.log(ROUND_up_csv / data)
        ans = ((j > logs) & (logs > -j)).replace({True: 1, False: 0})
        ans.to_csv(path_or_buf=f'{csvpath}/ROUND_up_out.csv')

                    
    def gen_down(): 
        
        logs = np.log(ROUND_down_csv / data)
        ans = ((j > logs) & (logs > -j)).replace({True: 1, False: 0})
        ans.to_csv(path_or_buf=f'{csvpath}/ROUND_down_out.csv')
    
    gen_up()
    gen_down()
    
def main(a, b, c, j):
    
    generate_MA(a, b, c, j)
    generate_TRB(a, b, c, j)
    generate_ROUND(j)