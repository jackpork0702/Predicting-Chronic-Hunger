# -*- coding: utf-8 -*-
'''
the tool for common data cleaning.
----------------------------------
This module require install matplotlib, seaborn, pandas, numpy and sklearn. 
'''


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing



# 有序類別轉換為0-1數值

def order_cat_to_num(df, tar_list, map_list):
    '''
    ----------------------------------
    from list to dictionary and make it's number between 0 to 1
    ----------------------------------
    df: the dataframe we are going to deal with 
    tar_list: the list contain target columns 
    map_list: what u wanna change to numeric by order(from less to great)
    ''' 
    map_dict = dict((k,map_list.index(k)/(len(map_list)-1)) for k in map_list)
    for tar in tar_list:
        df[tar] = df[tar].map(map_dict) 

         
            
            
        
# 空值為一類別

def NA_as_cat(df, tar_list, fill_with=-1):
    '''
    ----------------------------------
    NULL number is kind of category in some column
    ----------------------------------
    df: the dataframe we are going to deal with 
    tar_list: the list contain target columns 
    fill_with: what u wanna put inside null number
    ''' 
    for tar in tar_list:
        df.fillna({tar:fill_with}, inplace=True) 
        
        
            
        
        
# 標準化數值欄位(n倍標準差)

def standardize_col(df, tar_list):
    '''
    ----------------------------------
    standardize target columns
    ----------------------------------
    df: the dataframe we are going to deal with 
    tar_list: the list contain target columns 
    '''
    for tar in tar_list:
        df[tar] = preprocessing.scale(df[tar].values, copy=False)
        
      
    
        
        
# 類別資料攤平

def one_hot_encoding(df, tar_list, dummy_nan=True):
    '''
    ----------------------------------
    do one-hot encoding for category types' data,
    ----------------------------------
    df: the dataframe we are going to deal with 
    tar_list: the list contain target columns
    dummy_nan:  null will also take it as a category for default
    '''
    for tar in tar_list:
        df = pd.concat([df, pd.get_dummies(df[tar], prefix=tar, dummy_na=dummy_nan)], axis=1)
    df.drop(columns=tar_list, inplace=True)
    return df        
        
        
        
     
    
    
        
# 類別資料圖檢視

def cat_plot(df, tar, compare_with):
    '''
    ----------------------------------
    use plot to find out the relation with y
    ----------------------------------
    df: the dataframe we are going to deal with 
    tar: the column we want to compare with y
    compare_with: y
    '''
    
    # column's null number info
    df_null_info = pd.DataFrame(index = ['number', 'rate(%)'])
    unique_list = list(df[tar].unique())

    for i in range(len(unique_list)-1):
        df_null_info[unique_list[i]] = [
            (df[tar]==unique_list[i]).sum(),
            (df[tar]==unique_list[i]).sum()*100/len(df[tar])
        ]
    null_num = df[tar].isnull().sum()    
    df_null_info['null'] = [null_num,  null_num*100/len(df[tar])]
    df_null_info.iloc[1] = df_null_info.iloc[1].map(lambda x: ('%.2f')%x)
    print(df_null_info)
    
#     plot
    df_tmp = df.fillna('missing')                                           
    # x: tar,  y: count tar
    sns.countplot(tar, data=df_tmp)
    plt.show()
    # x: tar,  y: compare_with
    sns.catplot(data=df_tmp, x=tar, y=compare_with, kind='bar')
    plt.show()
    # x: tar,  y: compare_with
    sns.boxplot(data=df_tmp, x=tar, y=compare_with)
    plt.show()        
    

    
        

# 移除空值過多欄位

def drop_col(df_list, tar_list):
    '''
    ----------------------------------
    drop the columns don't need
    ----------------------------------
    df: the dataframe we are going to deal with 
    tar_list: the list contain target columns
    '''
    for df in df_list:
        df.drop(tar_list, inplace=True, axis=1)
        

        
        
# 將數據依等級轉化為整數
        
def persentage_level_transform (df, tar_list, quartile_list):
    '''
    ----------------------------------
    To transform data in persentage level. 
    Usually we use it to deal with the data which has many specific number, like '0'
    For example, we can transform the list [0,0,0,0,199,0,0,15896,0,0,0,22587,35,0,999999]
    into [0,0,0,0,1,0,0,2,0,0,0,3,1,0,4]. That can control some problem that may cause by outlier.
    ----------------------------------
    df: The pandas dataframe 
    tar_list: A list of the columns of df which are going to count
    quartile_list: [[], [], ...]
    This list can be made by 'data_describe.quartile_level_select(df, tar_list, n, beside_num)'.
    '''
    for num in range(len(tar_list)):
        col = tar_list[num]
        col_quartile_list = quartile_list[num]
        L = len(col_quartile_list)
        
        for i in range(L-1):
            df[col]=np.where(
                (df[col] >= col_quartile_list[i]) & (df[col] <= col_quartile_list[i+1]),
                i, df[col])
        df[col]=np.where(df[col] >= L, L-1, df[col])    

# if __name__ == '__main__':


