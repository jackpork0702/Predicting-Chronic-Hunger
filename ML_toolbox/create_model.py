# -*- coding: utf-8 -*-
"""
Here provide some simple method for ML model building.

r2_score:  http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
-----------------------------------
require modules: pandas, numpy, xgboost, matplotlib and sklearn 
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import time

class modeling():
    
    def __init__(self, df_x, df_y):
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(df_x, df_y, test_size=0.2)
        print("train_x", self.train_x.shape)
        print("train_y", self.train_y.shape)
        print("val_x", self.val_x.shape)
        print("val_y", self.val_y.shape)
    
    
    
    
    def linear_modeling(self,to_normalize ):
        '''
        -----------------------------------
        modeling.linear_modeling.coef_
        modeling.linear_modeling.intercept_
        
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        -----------------------------------
        to_normalize: normalize before model building (boolean)
        '''
        s = time.time()
        linear_regressor = LinearRegression(normalize=to_normalize)
        linear_model = linear_regressor.fit(self.train_x, self.train_y)
        
        pred_y = linear_model.predict(self.val_x)
        print('r2_score = ', r2_score(self.val_y, pred_y))
        print('cost time:', time.time()-s) 
        return linear_model
    
    
    
    
    def lasso_moedeling(self):
        '''
        -----------------------------------
        This method can also use Lasso to do feature selecting.
        Here provid a list call 'features' that can show the feature's coef isn't 0.
        
        LassoCV:  http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html
        Lasso:  http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        -----------------------------------
        '''
        s = time.time()
        lassocv = LassoCV()
        lassocv.fit(self.train_x, self.train_y)
        alpha = lassocv.alpha_
        print('alpha=', alpha)
        max_iter = lassocv.max_iter
        tol = lassocv.tol

        mask = (lassocv.coef_ != 0)
        features = list(self.train_x.loc[:,mask].columns)
        print('Feature select:\n', features)
        print('Features number', len(features))
        
        # create regressor
        lasso_regressor = Lasso(max_iter=max_iter, alpha=alpha, tol=tol)
        # create model
        lasso_model = lasso_regressor.fit(self.train_x, self.train_y)
        
        pred_y = lasso_model.predict(self.val_x)
        print('r2_score = ', r2_score(self.val_y, pred_y))
        
        print('cost time:', time.time()-s)
        return lasso_model
    
    
    
    
    def ridge_modeling(self, max_iterater=10000000):
        '''
        -----------------------------------
        Ridge:   http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        RidgeCV:  http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
        -----------------------------------
        '''
        s = time.time()
        ridgecv = RidgeCV()
        ridgecv.fit(self.train_x, self.train_y)
        alpha = ridgecv.alpha_
        # create regressor
        ridge_regressor = Ridge(max_iter=max_iterater, alpha=alpha)
        ridge_model = ridge_regressor.fit(self.train_x, self.train_y)
        
        pred_y = ridge_model.predict(self.val_x)
        print('r2_score = ', r2_score(self.val_y, pred_y))
        
        print('cost time:', time.time()-s)
        return ridge_model
    
    
    
    def elastic_modeling(self):
        '''
        -----------------------------------
        ElasticNetCV:  http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html
        ElasticNet:  http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
        -----------------------------------
        '''
        s = time.time()
        elasticnetcv = ElasticNetCV()
        elasticnetcv.fit(self.train_x, self.train_y)
        tol = elasticnetcv.tol
        l1_ratio = elasticnetcv.l1_ratio
        # create regressor
        elasticnet_regressor = ElasticNet(max_iter=10000000, l1_ratio=l1_ratio, tol=tol)
        # create model
        elasticnet_model = elasticnet_regressor.fit(self.train_x, self.train_y)
        
        pred_y = elasticnet_model.predict(self.val_x)
        print('r2_score = ', r2_score(self.val_y, pred_y))
        
        print('cost time:', time.time()-s)
        return elasticnet_model
    
    
    def rf_modeling(self, param_grid, n_jobs=-1, cv=5):
        '''
        -----------------------------------
        RandomForestRegressor: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        -----------------------------------
        param_grid: the condition for grid search.Example: 
                param_grid = { 
                    'bootstrap': [True],
                    'max_depth': [80, 90, 100, 110],
                    'max_features': [2, 3],
                    'min_samples_leaf': [3, 4, 5],
                    'min_samples_split': [8, 10, 12],
                    'n_estimators': [100, 200, 300, 1000]
                }
        n_jobs: Number of jobs to run in parallel. -1 means using all processors. default is -1. 
        cv: Determines the cross-validation splitting strategy. default is 5 (int)       
        '''
        s = time.time()
        rf = RandomForestRegressor()

        grid_rf = GridSearchCV(rf, param_grid, n_jobs=n_jobs, cv=cv)
        forest_model = grid_rf.fit(self.train_x, self.train_y)
        
        pred_y = forest_model.predict(self.val_x)
        print('Root Mean Square Error = ', str(math.sqrt(mean_squared_error(self.val_y, pred_y))))
        print('r2_score = ', r2_score(self.val_y, pred_y))
        
        print('cost time:', time.time()-s)
        return forest_model
    
    
    
    
    def xgb_modeling(self, param, num_round):
        '''
        -----------------------------------
        XGBoost; https://xgboost.readthedocs.io/en/latest/python/python_intro.html
        -----------------------------------
        param: the condition for xgb.train.
                Example:param = {
                                'objective':'reg:linear', # 做線性回歸
                                'tree_method':'hist',
                                'silent':1,
                                'max_depth':5
                                }
        num_round: num_boost_round (int) – Number of boosting iterations.
        '''
        s = time.time()
        # 將資料存成xgboost要求的型態
        data_val  = xgb.DMatrix(self.val_x, label=self.val_y)
        data_train = xgb.DMatrix(self.train_x, label=self.train_y)
                
        eval_list  = [(data_train,'train'),(data_val,'validation')]
#         num_round = 20
        
        eval_history={}

        # 訓練模型
        xgb_model = xgb.train(param, data_train, num_round, eval_list, 
                              evals_result=eval_history, verbose_eval=False)
    
        #檢視訓練情形
        rmse_train=eval_history['train']['rmse']
        rmse_validation=eval_history['validation']['rmse']
        plt.plot(rmse_train,ms=10,marker='.',label='train_eval')
        plt.plot(rmse_validation,ms=10,marker='v',label='validation_eval')
        plt.legend()
        plt.show()
    
        # 檢視最後rms error
        print("RMSE:", xgb_model.eval(data_val))
    
        #以R^2評估回歸結果(validation)
        pred_y = xgb_model.predict(data_val)
        print("r-square for validation data is", r2_score(self.val_y, pred_y))
    
        #以R^2評估回歸結果(train)
        train_pred_y = xgb_model.predict(data_train)
        print("r-square for train data is", r2_score(self.train_y, train_pred_y))
        
        print('cost time:', time.time()-s)
        return xgb_model