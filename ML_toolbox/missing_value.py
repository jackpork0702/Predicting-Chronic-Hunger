# -*- coding: utf-8 -*-
'''
Here provde some way for fill missing value. 
----------------------------------
This module require install fancyimpute, pandas, numpy and sklearn. 
Github of fnacyimpute: https://github.com/iskandr/fancyimpute
'''

import sys
import pandas as pd
import numpy as np
from fancyimpute import KNN, NuclearNormMinimization
from fancyimpute import SoftImpute, IterativeImputer, BiScaler
from sklearn.preprocessing import Imputer


def fill_mean(df):
    '''
    use mean to fill null number
    ----------------------------------
    df: the pandas.dataframe going to fill missing value
    '''
    fill_NaN = Imputer(missing_values=np.nan, strategy='median', axis=1)
    imputed_DF = pd.DataFrame(fill_NaN.fit_transform(df))
    imputed_DF.columns = df.columns
    imputed_DF.index = df.index
    
    return imputed_DF


def fill_knn(df, neighbourhood=5):
    '''
    use KNN to fill null number
    ----------------------------------
    df: the pandas.dataframe going to fill missing value
    neighbourhood: KNN neighbour number
    '''
    df_filled_knn =pd.DataFrame(KNN(k=neighbourhood).fit_transform(df.as_matrix()))
    df_filled_knn.columns = df.columns
    df_filled_knn.index = df.index

    return df_filled_knn


def fill_ii(df):
    '''
    Use IterativeImputer to fill null number
    ----------------------------------
    df: the pandas.dataframe going to fill missing value
    '''
    df_filled_ii = pd.DataFrame(IterativeImputer().fit_transform(df.as_matrix()))
    df_filled_ii.columns = df.columns
    df_filled_ii.index = df.index

    return df_filled_ii