#!/usr/bin/env python
# coding: utf-8

# In[92]:


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
from sympy import symbols, simplify, lambdify, Function, diff, Mul
from sklearn.metrics import mean_squared_error
from cvxopt import matrix, solvers
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix
import math
from scipy.stats import pearsonr
d = int(input("Enter polynomial kernel d = "))
c = int(input("Enter c =  "))
def compute_kernel_matrix(X, c, d):
    n = X.shape[0]
    K = (c + np.dot(X, X.T)) ** d
    return K
def solve_for_lambda(data, c, d, lambda_value):
    B501_data = pd.read_csv('../../../Data/INTEGRAL_MANIFOLDS/Exam7_T5_N100/B501.csv')
    traj_len = B501_data.groupby('trajectory').size()
    rep = int(round(traj_len.mean()))
    K = compute_kernel_matrix(data, c, d)
    I = np.eye(K.shape[0])
    K_with_I = K + lambda_value * I
    K_with_I_inv = np.linalg.inv(K_with_I)
    m = 5 # number of trajectory
    num_y_variables = m
    y_symbols = [sp.symbols(f'y{i}') for i in range(num_y_variables)]
    y_pattern = [sp.symbols(f'y{i}') for i in range(m)]
    y_repeated = np.repeat(y_pattern, rep, axis=0)
    y = sp.Matrix([sp.symbols(f'y{i}') for i in range(m)])  
    M_matrix = sp.Matrix(K_with_I_inv)
    n = K.shape[0]
    M = sp.zeros(n, m)
    for i in range(n):
        for j in range(m):
            start_idx = j * rep
            end_idx = (j + 1) * rep
            M[i, j] = sp.Add(*M_matrix[i, start_idx:end_idx])
    M_transpose = M.transpose()
    A = M_transpose @ M 
    A_np = np.array(A).astype(np.float64)
    return A_np
m = 5 # number of trajectory
base_path = "../../../Data/INTEGRAL_MANIFOLDS/Exam7_T5_N100/"
file_names = [f"B50{i}.csv" for i in range(1, m+1)]
filenames = [base_path + file_name for file_name in file_names]
data_list = [np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(0, 1, 2)) for filename in filenames]
data, data2, data3, data4, data5 = data_list
solutions_combined = []
for i in range(0, 8):
    lambda_value = 10**(3-i)
    A_np_list = [solve_for_lambda(data_array, c=c, d=d, lambda_value=lambda_value) for data_array in data_list]
    A1_np, A2_np, A3_np, A4_np, A5_np = A_np_list
    As_p = np.mean(A_np_list, axis=0)
    P = np.array(As_p)
    q = np.zeros(m)
    a = np.random.uniform(-3, 3, m)
    G = np.zeros((0, m))
    h = np.zeros(0)
    A = a.reshape(1, -1)
    P = csc_matrix(P)
    G = csc_matrix(G)
    A = csc_matrix(A) 
    b = np.array([1.0])
    y = solve_qp(P, q, G, h, A, b, solver="clarabel")
    y_opt = y.flatten() if isinstance(y, np.ndarray) else np.array(y).flatten()
    y_names = [sp.symbols(f'y{i}') for i in range(m)]
    y_dict = {name: value for name, value in zip(y_names, y_opt)}
    solutions_combined.append((lambda_value, y_dict))
def compute_kernel_matrix(X, c, d):
    n = X.shape[0]
    K = (c + np.dot(X, X.T)) ** d
    return K
def solve_for_lambda(data, c, d, lambda_value, y_values):
    K = compute_kernel_matrix(data, c, d)
    I = np.eye(K.shape[0])
    K_with_I = K + lambda_value * I
    K_with_I_inv = np.linalg.inv(K_with_I)
    y_repeated = np.repeat(y_values, len(data) // len(y_values))
    alpha_sym = K_with_I_inv @ y_repeated
    return K_with_I_inv, alpha_sym
def generate_f_alpha_expression(alpha_sym, x1, x2, x3, x_q1_sym, x_q2_sym, x_q3_sym, c, d):
    f_alpha = 0
    for i in range(len(alpha_sym)):
        f_alpha += alpha_sym[i] * (c + (x1[i] * x_q1_sym) + (x2[i] * x_q2_sym) + (x3[i] * x_q3_sym)) ** d
    f_alpha_expanded = sp.expand(f_alpha)
    f_alpha_collected = sp.collect(f_alpha_expanded, (x_q1_sym, x_q2_sym, x_q3_sym))
    return f_alpha_collected
def process_dataset(file_path, c, d, lambda_values, y_values_dicts):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(0, 1, 2))
    x1, x2, x3 = data[:, 0], data[:, 1], data[:, 2]
    K_with_I_inv_list = []
    alpha_sym_list = []
    f_alpha_expression_list = []
    for lambda_val, (_, y_values_dict) in zip(lambda_values, y_values_dicts):
        K_with_I_inv, alpha_sym = solve_for_lambda(data, c, d, lambda_val, list(y_values_dict.values()))
        x_q1_sym, x_q2_sym, x_q3_sym = sp.symbols('x1 x2 x3')
        K_with_I_inv_list.append(K_with_I_inv)
        alpha_sym_list.append(alpha_sym)
        f_alpha_expression = generate_f_alpha_expression(alpha_sym, x1, x2, x3, x_q1_sym, x_q2_sym, x_q3_sym, c, d)
        f_alpha_expression_list.append(f_alpha_expression)
    
    return K_with_I_inv_list, alpha_sym_list, f_alpha_expression_list
lamda = 8
lambda_values = [10**(3-i) for i in range(lamda)]
y_values_dicts_list = solutions_combined
base_path = "../../../Data/INTEGRAL_MANIFOLDS/Exam7_T5_N100/"
file_names = [f"B50{i}.csv" for i in range(1, 6)]
file_paths = [base_path + file_name for file_name in file_names]
datasets = [pd.read_csv(path) for path in file_paths]
B501_data = datasets[0]
dr1 = datasets[-1]
traj_len = B501_data.groupby('trajectory').size()
rep4 = int(round(traj_len.mean()))
results = [process_dataset(path, c, d, lambda_values, y_values_dicts_list) for path in file_paths]
f_vectors = []
for _, _, f_alpha_expr in results:
    f_vector = [[] for _ in range(len(f_alpha_expr))]
    for index, row in dr1.iterrows():
        for i, expr in enumerate(f_alpha_expr):
            value = eval(str(expr), globals(), row.to_dict())
            f_vector[i].append(value)
    f_vectors.append(f_vector)
h_values = [list(y_dict[1].values()) for y_dict in y_values_dicts_list]
y_B_values = []
for h in h_values:
    y_B_values.extend([np.repeat(h, rep4) for _ in range(8)])
rmse_values = {}
for i, f_vector in enumerate(f_vectors):
    rmse_values[f"B50{i+1}"] = [np.sqrt(mean_squared_error(y, f)) for y, f in zip(y_B_values[i*8:(i+1)*8], f_vector)]
min_rmse_info = {}
for key, rmse_list in rmse_values.items():
    min_rmse = min(rmse_list)
    min_index = rmse_list.index(min_rmse)
    min_rmse_info[key] = (min_rmse, min_index)
min_rmse = float('inf')  
min_index = None
dataset_key = None
for key, rmse_list in rmse_values.items():
    for index, value in enumerate(rmse_list):
        if value < min_rmse:
            min_rmse = value
            min_index = index
            dataset_key = key
file_index = min_index // 8 + 1
sub_index = min_index % 8
lambda_values = [10**(3-i) for i in range(lamda)]
h_value = h_values[-1]
df2 = pd.read_csv('../../../Data/INTEGRAL_MANIFOLDS/Exam7_T5_N100/trainingp_data50.csv')
m = df2['trajectory'].nunique()
df2['trajectory'] = df2['trajectory'].replace({i: h_value[i-1] for i in range(1, m+1)})
X_train = df2.iloc[:, :-1]
y_train = df2.iloc[:, -1]
X_train.to_csv('../../../results/INTEGRAL_MANIFOLDS/Exam7_T5_N100/X_train.csv', index=False)
y_train = y_train.astype(float)
X_train = X_train.astype(float)
def polynomial_kernel(X, Y, degree=d):
    return (1 + np.dot(X, Y.T)) ** degree
param_grid = {'alpha': [0.00002, 0.004, 0.06, 0.1, 1, 10, 100, 1000]}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
kr_model = KernelRidge(kernel=polynomial_kernel)
grid_search = GridSearchCV(kr_model, param_grid, cv=cv, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# print("Best RMSE:", -grid_search.best_score_)
# print("")
# print("")
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
x1, x2, x3 = sp.symbols('x1 x2 x3') 
polynomial_kernel = (1 + x1*sp.Symbol('xi1') + x2*sp.Symbol('xi2') + x3*sp.Symbol('xi3'))**d
f_beta = 0
for i in range(len(X_train)):
    f_beta += eta[i] * polynomial_kernel.subs({'xi1': X_train.iloc[i][0], 'xi2': X_train.iloc[i][1], 'xi3': X_train.iloc[i][2]})
candidate_CL = sp.expand(f_beta)
print("Candidate Conservation Law:")
sp.pprint(candidate_CL)
print("")
print("")
coefficients = list(candidate_CL.as_coefficients_dict().values())
terms = list(candidate_CL.as_coefficients_dict().keys())
filtered_terms = [term for coeff, term in zip(coefficients, terms) if abs(coeff) > 0.0001]
filtered_ex = sum(Mul(coeff, term) for coeff, term in zip(coefficients, terms) if term in filtered_terms)
print("Final Conservation Law:")
sp.pprint(filtered_ex)
print("")
print("")
with open("../../../results/INTEGRAL_MANIFOLDS/Exam7_T5_N100/candidate_CL.txt", "w") as file:
    file.write(str(candidate_CL))
with open("../../../results/INTEGRAL_MANIFOLDS/Exam7_T5_N100/candidate_CL.txt", "r") as file:
    candidate_CL = sp.sympify(file.read())
df3 = pd.read_csv('../../../Data/INTEGRAL_MANIFOLDS/Exam7_T5_N100/holdoutp_data50.csv')
traj_len = df3.groupby('trajectory').size()
rep1 = int(round(traj_len.mean()))
expression = sp.lambdify((x1, x2, x3), candidate_CL, "numpy")
df3['lamhold'] = expression(df3['x1'], df3['x2'], df3['x3'])
da = {'y{}'.format(i): h_value[i] for i in range(len(h_value))}
df3['Coluh(lamhold)'] = [da[f'y{i}'] for i in range(m) for _ in range(rep1)]
columns_to_compare = [('lamhold', 'Coluh(lamhold)')]
for col1, col2 in columns_to_compare:
    rmse = np.sqrt(mean_squared_error(df3[col1], df3[col2]))
    print(f'Generalisation Error (RMSE): {rmse}')
    print("")
with open("../../../results/INTEGRAL_MANIFOLDS/Exam7_T5_N100/candidate_CL.txt", "r") as file:
    candidate_CL = sp.sympify(file.read())
f = sp.lambdify((x1, x2, x3), candidate_CL, "numpy")
dat = pd.read_csv('../../../Data/INTEGRAL_MANIFOLDS/Exam7_T5_N100/holdoutp_data50.csv')
trajectories = dat['trajectory'].unique()
total_sum_squared_normalized_functional_value = 0
total_data_points = 0
num_x_variables = 3
for trajectory in trajectories:
    trajectory_data = dat[dat['trajectory'] == trajectory].copy()  
    cols = ['x' + str(i) for i in range(1, num_x_variables + 1)] # number of variable
    trajectory_data['functional_value'] = f(*trajectory_data[cols].values.T)
    mean_value = trajectory_data['functional_value'].mean()
    trajectory_data['functional_value_minus_mean'] = trajectory_data['functional_value'] - mean_value
    trajectory_data['normalized_functional_value'] = trajectory_data['functional_value_minus_mean'] / mean_value
    trajectory_data['squared_normalized_functional_value'] = trajectory_data['normalized_functional_value'] ** 2
    total_sum_squared_normalized_functional_value += trajectory_data['squared_normalized_functional_value'].sum()
    total_data_points += len(trajectory_data)
average_squared_normalized_functional_value = total_sum_squared_normalized_functional_value / total_data_points
standard_deviation = math.sqrt(average_squared_normalized_functional_value)
print(" Relative deviation:", standard_deviation)
print("")


# In[ ]:





# In[ ]:





# In[ ]:




