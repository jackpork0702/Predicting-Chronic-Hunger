# -*- coding: utf-8 -*-
'''
here provide the way to deal with data in numeric type outlier.
----------------------------------
This module require install pandas and numpy. 
pip install pandas
pip install numpy
'''


import pandas as pd
import numpy as np
import sys


def filter_extreme_Nsigma(df, tar_list, n=3):
    '''
    ----------------------------------
    convert outlier by N times sigma.

    if x >= mean + n*sigma, then x = mean + n*sigma
    if x <= mean - n*sigma, then x = mean - n*sigma
    
    the data should be normalie or be normal distrubution first.
    ----------------------------------
    df: target dataframe
    tar_list: the list you want to convert outlier
    n: N times sigma
    '''
    df_new = pd.DataFrame()
    for col in tar_list:
        col_mean = df[col].mean()
        col_std = df[col].std()
        max_range = col_mean + n*col_std
        min_range = col_mean - n*col_std
        df_new[col] = df[col].clip(min_range, max_range)
    return df_new



def Tukey_test(df, tar_list, k=3):
    '''
    ----------------------------------
    convert outlier by k times IQR.
    
    if x >= Q3 + k*IQR, then x = Q3 + k*IQR
    if x <= Q1 - k*IQR, then x = Q1 - k*IQR
    ----------------------------------
    df: target dataframe
    tar_list: the list you want to convert outlier
    k: k times IQR
    '''
    df_new = pd.DataFrame()
    for col in tar_list:
        q1 = np.percentile(df[col].values, 25)
        q3 = np.percentile(df[col].values, 75)
        IQR = q3 - q1
        max_range = q3 + k*IQR
        min_range = q1 - k*IQR
        df_new[col] = df[col].clip(min_range, max_range)
    return df_new




# if __name__ == '__main__':
    