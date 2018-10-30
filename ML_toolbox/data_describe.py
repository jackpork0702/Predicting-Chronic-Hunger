# -*- coding: utf-8 -*-
'''
This tool provides an easy and quick over view for data. 
It helps user to select null features. 
----------------------------------
This module require install pandas and numpy. 
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
'''


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


def data_describe(df):
    '''
    ----------------------------------
    'duplicate' column is the number of each feature in data which contain how many different members.
    'is_null' column means whether the feature has null number or not.
    'null_number' means how many null numbers does the feature has.
    ''null_rate(%)' is the persentage show that feature has. Be calculated like: null_number/len(df[feature])
    ----------------------------------
    df: the data frame user want to check.
    '''
#     df.info()
#     df.shape()
#     print(df.isnull().any().sum(), "columns have null number")

    describe = pd.DataFrame()
    _duplicate_num(df, describe)
    _is_null_number(df, describe)
    describe['null_rate(%)'] = describe['null_number']*100/len(df)
    describe['null_rate(%)'] = describe['null_rate(%)'].map(lambda x: ('%.2f')%x)

    return describe
    
    
    
def _duplicate_num(df, describe):
    dup_num = df.apply(lambda x:x.unique().shape[0], axis=0)
    describe['duplicate'] = dup_num
    
    
    
def _is_null_number(df, describe):
    describe['is_null'] = df.isnull().any()
    describe['null_number'] = df.isnull().sum()

    
    
def data_type (describe, default_type_name, data_type_dict):
    '''
    ----------------------------------
    Append a column 'type' behind the table from data_describe(df).
    Put a dictionary on  'data_type_dict'. 
    Beside the item in 'data_type_dict' will be setted as 'default_type_name'.
    ----------------------------------
    describe: This is the pandas dataframe from 'data_describe(df)'.
    default_type_name: The type will be set for the feature beside the item you set in 'data_type_dict'.            
    data_type_dict:  
            Example:  data_type_dict = {'category':['month', 'type'], 'numeric':['price','quantity']}
    '''
    describe['type'] = default_type_name
    for type_name in data_type_dict:
        describe.loc[data_type_dict[type_name],'type'] = type_name
    
    

    
def quartile_level_select (df, tar_list, n=4, beside_num=None):
    '''
    ----------------------------------
    To count quartile of dataframe's columns beside 'beside_num'
    thew result that this funtion return will be a list include many list.
    the sub lists are quartile info of each item in target list.
    ----------------------------------
    df: The pandas dataframe 
    tar_list: A list of the columns of df which are going to count
    n: How many levels are going to count quartile
            Example: n=4 -> return a list [0%, 25%, 50%, 75%, 100%]
    beside_num: default is None that means normal way to count quartile. 
                If put a number here that means to count quartile beside this number.
    '''
    N = 100/n
    Q=[]
    for item in tar_list:
        persentage = 0
        q = [0 for i in range(n+1)]
        A = df[df[item]!=beside_num][item]
        for i in range(n):
            persentage += N
            q[i+1] = np.percentile(A.values, persentage)
        Q.append(q)  
    return Q
    
    
    
def null_pic(df):
    '''
    ----------------------------------
    A picture show scatter of null number  
    ----------------------------------
    df: The pandas.DataFrame user want to show in picture 
    '''
    plt.figure(figsize=(20,10))
    cmap=sns.light_palette("navy", reverse=False)
    sns.heatmap(df.isnull().astype(np.int8),yticklabels=False,cmap=cmap)
    plt.show()


def duplicate_pic(df):
    '''
    ----------------------------------
    A picture show duplicate number  
    ----------------------------------
    df: The pandas.DataFrame user want to show in picture    
    '''
    dup_num = df.apply(lambda x:x.unique().shape[0], axis=0)
    plt.figure(figsize=(20,10))
    (dup_num/df.shape[0]).plot(kind='bar', rot=75, title='duplicate number rate')
    plt.show()
    
    
    
# if __name__ == '__main__':

    