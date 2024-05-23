#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import csv
import itertools
import os
import random
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, KFold
from sklearn.kernel_ridge import KernelRidge
from sympy import symbols, simplify, lambdify, Function, diff
from sklearn.metrics import mean_squared_error
import math
from scipy.stats import pearsonr
# get_ipython().run_line_magic('matplotlib', 'inline')
dm = pd.read_csv("../../../System/Implicitisation/Trifolium/50.csv")
x_columns = [col for col in dm.columns if col.startswith('x')]
num_x_variables = len(x_columns)
d = int(input("Enter degree for polynomial kernel: "))
df2 = pd.read_csv('../../../System/Implicitisation/Trifolium/50.csv')
X = df2.iloc[:, :-1]
y = df2.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.to_csv('../../../results/Implicitisation/Trifolium/X_train.csv', index=False)
y_train = y_train.astype(float)
def polynomial_kernel(X, Y, degree=d):
    return (1 + np.dot(X, Y.T)) ** degree
param_grid = {'alpha': [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
kr_model = KernelRidge(kernel=polynomial_kernel)
grid_search = GridSearchCV(kr_model, param_grid, cv=cv, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best RMSE:", -grid_search.best_score_)
print("")
print("")
grid_search.best_estimator_
y_pred = grid_search.predict(X_test)
class KernelMethodBase(object):
    '''
    Base class for kernel methods models
    Methods
    ----
    fit
    predict
    fit_K
    predict_K
    '''
    kernels_ = {
        'polynomial': polynomial_kernel,
    }
    def __init__(self, kernel='polynomial', **kwargs):
        self.kernel_name = kernel
        self.kernel_function_ = self.kernels_[kernel]
        self.kernel_parameters = self.get_kernel_parameters(**kwargs)
        self.fit_intercept_ = False
    def get_kernel_parameters(self, **kwargs):
        params = {}
        params['degree'] = kwargs.get('degree', d)
        return params
    def fit_K(self, K, y, **kwargs):
        pass
    def decision_function_K(self, K):
        pass
    def fit(self, X, y, **kwargs):
        self.X_train = X
        self.y_train = y
        K = self.kernel_function_(self.X_train, self.X_train, **self.kernel_parameters)
        return self.fit_K(K, y, **kwargs)
    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)
        return self.decision_function_K(K_x)
    def predict(self, X):
        pass
    def predict_K(self, K):
        pass
class KernelRidgeRegression(KernelMethodBase):
    '''
    Kernel Ridge Regression
    '''
    def __init__(self, alpha=0.1, **kwargs):
        self.alpha = alpha
        super(KernelRidgeRegression, self).__init__(**kwargs)
    def fit_K(self, K, y):
        n = K.shape[0]
        assert (n == len(y))
        A = K + self.alpha*np.identity(n)
        self.eta = np.linalg.solve(A , y)
        return self
    def decision_function_K(self, K_x):
        return K_x.dot(self.eta)
    def predict(self, X):
        return self.decision_function(X)
    def predict_K(self, K_x):
        return self.decision_function_K(K_x)
kernel = 'polynomial'
kr_model = KernelRidgeRegression(
    kernel=kernel,
    alpha=grid_search.best_params_['alpha'],
    )
kr_model.fit(X_train, y_train)
eta = kr_model.eta
x1, x2 = sp.symbols('x1 x2')
polynomial_kernel = (1 + x1*sp.Symbol('xi1') + x2*sp.Symbol('xi2'))**d
f_beta = 0
for i in range(len(X_train)):
    f_beta += eta[i] * polynomial_kernel.subs({'xi1': X_train.iloc[i][0], 'xi2': X_train.iloc[i][1]})
expanded_f_beta = sp.expand(f_beta)
print("Candidate Conservation Law Phi(x):")
sp.pprint(expanded_f_beta)
print("")
print("")
coef = list(expanded_f_beta.as_coefficients_dict().values())
terms = list(expanded_f_beta.as_coefficients_dict().keys())
filtered_terms = [term for coeff, term in zip(coef, terms) if abs(coeff) > 0.0001]
filtered = sum(sp.Mul(coeff, term) for coeff, term in zip(coef, terms) if term in filtered_terms)
print("Final Candidate CL:")
sp.pprint(filtered)


# In[ ]:





# In[ ]:




