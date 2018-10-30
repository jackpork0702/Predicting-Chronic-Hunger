# -*- coding: utf-8 -*-
'''
This tool provides tools to select features by correlation. 
----------------------------------
This module require install pandas and numpy. 
pip install pandas
pip install numpy
'''

import pandas as pd
import numpy as np
import sys


def multicollinearity_prevent(df, y_lab, corr_method = 'spearman', level=0.7):
    '''
    ----------------------------------
    prevent multicollinearity. 
    if the features pair reach the level, compare their correlation value with y.
    Then drop the lower one. 
    ----------------------------------
    df: target dataframe contain y 
    tar: y
    corr_method: the method of correlation calculate (‘pearson’, ‘kendall’, ‘spearman’), default 'spearman' 
    level: the level u want to select to remove feature, default 0.7
    '''
    corr_table = df.corr(method = corr_method)
    
    left_col = list(corr_table.index)
    drop_list = []

    for row in left_col:
        for col in  left_col:
            if abs(corr_table[row][col]) > level:
                drop_list.append([row, col])

    drop_col = []
    for pair_list in drop_list:
        if pair_list[0] == pair_list[1]:     # skip diagonal, because it always be 1
            pass
        else: 
            corr_A = corr_table[y_lab][pair_list[0]]
            corr_B = corr_table[y_lab][pair_list[1]]
            if corr_A > corr_B: drop_col.append(pair_list[0])
            else: drop_col.append(pair_list[1])

#     return df.drop(columns=set(drop_col))
    return set(drop_col)




def corr_select(df, y_lab, corr_method, level):
    '''
    ----------------------------------
    select the features which reach the level have set.
    ----------------------------------
    df: target dataframe contain y 
    y_lab: y label
    corr_method: the method of correlation calculate (‘pearson’, ‘kendall’, ‘spearman’), default 'spearman' 
    level: the level you want to select to remove feature
    '''
    drop_list=[]
    for i in df.columns[2:]:   # beside row_id and country_code
        if abs(df[y_lab].corr(df[i], method = corr_method)) < level:
            drop_list.append(i)

    df.drop(drop_list, axis = 1, inplace=True)
    print(df.shape)
    print(df.columns)
    
    
    
# if __name__ == '__main__':


