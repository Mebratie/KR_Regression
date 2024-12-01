#!/usr/bin/env python
# coding: utf-8

# Author: Meskerem Abebaw Mebratie
# Date: 2024-12-01

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
from itertools import combinations_with_replacement
from sympy import symbols, Matrix
from scipy import sparse
import ast
import cvxpy as cp
d = int(input("Enter polynomial kernel d = "))
c = float(input("Enter c = "))
def compute_kernel_matrix(X, c, d):
    n = X.shape[0]
    K = (c + np.dot(X, X.T)) ** d
    return K
def solve_for_lambda(data, c, d, lambda_value):
    B501_data = pd.read_csv('../../../Data/Lorenz_System/Lorenz_System_T5_40/B501.csv')
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
base_path = "../../../Data/Lorenz_System/Lorenz_System_T5_40/"
file_names = [f"B50{i}.csv" for i in range(1, m+1)]
filenames = [base_path + file_name for file_name in file_names]
data_list = [np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4)) for filename in filenames]
data, data2, data3, data4, data5 = data_list
solutions_combined = []
for i in range(0, 8):
    lambda_value = 10**(-i)
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
def generate_f_alpha_expression(alpha_sym, x1, x2, x3, x4, x5, x_q1_sym, x_q2_sym, x_q3_sym, x_q4_sym, x_q5_sym, c, d):
    f_alpha = 0
    for i in range(len(alpha_sym)):
        f_alpha += alpha_sym[i] * (c + (x1[i] * x_q1_sym) + (x2[i] * x_q2_sym) + (x3[i] * x_q3_sym) + (x4[i] * x_q4_sym) + (x5[i] * x_q5_sym)) ** d
    f_alpha_expanded = sp.expand(f_alpha)
    f_alpha_collected = sp.collect(f_alpha_expanded, (x_q1_sym, x_q2_sym, x_q3_sym, x_q4_sym, x_q5_sym))
    return f_alpha_collected
def process_dataset(file_path, c, d, lambda_values, y_values_dicts):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4))
    x1, x2, x3, x4, x5 = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    K_with_I_inv_list = []
    alpha_sym_list = []
    f_alpha_expression_list = []
    for lambda_val, (_, y_values_dict) in zip(lambda_values, y_values_dicts):
        K_with_I_inv, alpha_sym = solve_for_lambda(data, c, d, lambda_val, list(y_values_dict.values()))
        x_q1_sym, x_q2_sym, x_q3_sym, x_q4_sym, x_q5_sym = sp.symbols('x1 x2 x3 x4 x5')
        K_with_I_inv_list.append(K_with_I_inv)
        alpha_sym_list.append(alpha_sym)
        f_alpha_expression = generate_f_alpha_expression(alpha_sym, x1, x2, x3, x4, x5, x_q1_sym, x_q2_sym, x_q3_sym, x_q4_sym, x_q5_sym, c, d)
        f_alpha_expression_list.append(f_alpha_expression)
    
    return K_with_I_inv_list, alpha_sym_list, f_alpha_expression_list
lamda = 8
lambda_values = [10**(-i) for i in range(lamda)]
y_values_dicts_list = solutions_combined
base_path = "../../../Data/Lorenz_System/Lorenz_System_T5_40/"
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
# h_values = [h1, h2, h3, h4, h5, h6, h7, h8]
lambda_values = [10**(-i) for i in range(lamda)]
h_value = h_values[-1]
# h_value = h_values[sub_index]
df2 = pd.read_csv('../../../Data/Lorenz_System/Lorenz_System_T5_40/trainingp_data50.csv')
m = df2['trajectory'].nunique()
df2['trajectory'] = df2['trajectory'].replace({i: h_value[i-1] for i in range(1, m+1)})
X_train = df2.iloc[:, :-1]
y_train = df2.iloc[:, -1]
X_train.to_csv('../../../results/Lorenz_System/Lorenz_System_T5_40/X_train.csv', index=False)
y_train = y_train.astype(float)
X_train = X_train.astype(float)
def polynomial_kernel(X, Y, degree=d):
    return (1 + np.dot(X, Y.T)) ** degree
param_grid = {'alpha': [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
kr_model = KernelRidge(kernel=polynomial_kernel)
grid_search = GridSearchCV(kr_model, param_grid, cv=cv, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
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
x1, x2, x3, x4, x5 = sp.symbols('x1 x2 x3 x4 x5') 
polynomial_kernel = (1 + x1*sp.Symbol('xi1') + x2*sp.Symbol('xi2') + x3*sp.Symbol('xi3') + x4*sp.Symbol('xi4') + x5*sp.Symbol('xi5'))**d
f_beta = 0
for i in range(len(X_train)):
    f_beta += eta[i] * polynomial_kernel.subs({'xi1': X_train.iloc[i][0], 'xi2': X_train.iloc[i][1], 'xi3': X_train.iloc[i][2], 'xi4': X_train.iloc[i][3], 'xi5': X_train.iloc[i][4]})
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
exp1 = filtered_ex
div1 = exp1
simp_exp1 = simplify(div1)
filtered_exp1 = sum(term for term in simp_exp1.args if term.has(x1) or term.has(x2) or term.has(x3))
print("Candidate CL:")
sp.pprint(filtered_exp1)
print("")
print("")
exp = filtered_ex
tar = x5 * x3
tar_coef = exp.coeff(tar)
deno = tar_coef
div = exp / deno
simp_exp = simplify(div)
filtered_exp = sum(term for term in simp_exp.args if term.has(x1) or term.has(x2) or term.has(x3))
print("Simplified Candidate CL:")
sp.pprint(filtered_exp)
print("")
print("")
with open("../../../results/Lorenz_System/Lorenz_System_T5_40/candidate_CL.txt", "w") as file:
    file.write(str(candidate_CL))
with open("../../../results/Lorenz_System/Lorenz_System_T5_40/candidate_CL.txt", "r") as file:
    candidate_CL = sp.sympify(file.read())
df3 = pd.read_csv('../../../Data/Lorenz_System/Lorenz_System_T5_40/holdoutp_data50.csv')
traj_len = df3.groupby('trajectory').size()
rep1 = int(round(traj_len.mean()))
expression = sp.lambdify((x1, x2, x3, x4, x5), candidate_CL, "numpy")
df3['lamhold'] = expression(df3['x1'], df3['x2'], df3['x3'], df3['x4'], df3['x5'])
da = {'y{}'.format(i): h_value[i] for i in range(len(h_value))}
df3['Coluh(lamhold)'] = [da[f'y{i}'] for i in range(m) for _ in range(rep1)]
columns_to_compare = [('lamhold', 'Coluh(lamhold)')]
for col1, col2 in columns_to_compare:
    rmse = np.sqrt(mean_squared_error(df3[col1], df3[col2]))
    print(f'Generalisation Error (RMSE): {rmse}')
    print("")
with open("../../../results/Lorenz_System/Lorenz_System_T5_40/candidate_CL.txt", "r") as file:
    candidate_CL = sp.sympify(file.read())
f = sp.lambdify((x1, x2, x3, x4, x5), candidate_CL, "numpy")
dat = pd.read_csv('../../../Data/Lorenz_System/Lorenz_System_T5_40/holdoutp_data50.csv')
trajectories = dat['trajectory'].unique()
total_sum_squared_normalized_functional_value = 0
total_data_points = 0
num_x_variables = 5
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

##### Second Search

def solve_for_lambda(data, c, d, lambda_value):
    B5d = pd.read_csv('../../../Data/Lorenz_System/Lorenz_System_T5_40/trainingp_data50.csv')
    traj_len = B5d.groupby('trajectory').size()
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
    A = M 
    F =A.T
    return A, K_with_I_inv, F
data = np.loadtxt('../../../Data/Lorenz_System/Lorenz_System_T5_40/trainingp_data50.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4))
solutions_combined = []
for i in range(7, 8):
    lambda_value = 10**(-i)
    A, K_with_I_inv, F = solve_for_lambda(data, c=1, d=2, lambda_value=lambda_value)
def multilinear_coefficient(n, *ks):
    numerator = math.factorial(n)
    denominator = 1
    for k in ks:
        denominator *= math.factorial(k)
    return numerator // denominator
def multilinear_expansion(variables, n, row):
    expansions = []
    for ks in itertools.product(range(n + 1), repeat=len(variables)):
        if sum(ks) == n:
            coefficient = multilinear_coefficient(n, *ks)
            values = [row[var] ** k if var != '1' else 1 for var, k in zip(variables, ks)]
            term_value = coefficient * math.prod(values)
            expansions.append(term_value)
    return expansions[::-1]
def multilinear_expansion1(variables1, n1):
    expansions1 = []  
    for ks in itertools.product(range(n1 + 1), repeat=len(variables1)):
        if sum(ks) == n1:
            terms1 = [f"{var}**{k}" if k != 0 else f"{var}" for var, k in zip(variables1, ks) if k != 0]
            term1 = " * ".join(terms1)
            expansions1.append(term1)
    return expansions1[::-1]
def generate_inner_products(coefficients, terms):
    inner_products = [f"{c}*{t}" for c, t in zip(coefficients, terms)]
    return inner_products
data = pd.read_csv('../../../Data/Lorenz_System/Lorenz_System_T5_40/trainingp_data50.csv')
n = 2
data['c'] = 1
num_x_variables = 5
variables = ['c'] + [f'x{i}' for i in range(1, num_x_variables + 1)]
n1 = 2
variables1 = ['1'] + [f'x{i}' for i in range(1, num_x_variables + 1)]
all_entries = []
for index, row in data.iterrows():
    result = multilinear_expansion(variables, n, row)
    result1 = multilinear_expansion1(variables1, n1)
    expressions = generate_inner_products(result, result1)
    all_entries.append(result)
matrix = np.array(all_entries)
def generate_symbolic_terms(degree=2):
    terms = []
    variables = [x1, x2, x3, x4, x5]
    terms.append('1**1')
    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(variables, d):
            term = ' * '.join([f'{v}**{comb.count(v)}' for v in set(comb)])
            terms.append(term)
    return terms
def print_as_single_row(terms):
    matrix_row = "  ".join(terms)
    shape = (1, len(terms)) 
result = generate_symbolic_terms(degree=2)
print_as_single_row(result)
F_numpy = np.array(F.tolist()).astype(np.float64)
if matrix.shape[1] == F_numpy.shape[0]:
    result = np.dot(F_numpy, matrix)
ds=np.dot(F_numpy, matrix)
shape = print_as_single_row(result)
symbolic_matrix = sp.Matrix(1, len(result), result)
ds_symbolic = ds * symbolic_matrix.T 
shape = print_as_single_row(result)
symbolic_matrix = sp.Matrix(1, len(result), result)
ds_symbolic = ds * symbolic_matrix.T 
def compute_gradients(expr, variables):
    return [sp.diff(expr, var) for var in variables]
variables = [x1, x2, x3, x4, x5]
gradient_rows = []
for i in range(ds_symbolic.shape[0]):
    gradient_row = compute_gradients(ds_symbolic[i, 0], variables)
    gradient_rows.append(gradient_row)
gradients_matrix = sp.Matrix(gradient_rows)
N = 1
D = num_x_variables
lower_bound = -2
upper_bound = 2
points = np.random.uniform(lower_bound, upper_bound, (N, D))
rp = ', '.join(f"{x:.8f}" for x in points[0])
values = [float(x) for x in rp.split(', ')]
x1_value, x2_value, x3_value, x4_value, x5_value = values
P_J = gradients_matrix.subs({x1: x1_value, x2: x2_value, x3: x3_value, x4: x4_value, x5: x5_value})
P_J_matrix = Matrix(P_J)
rank_P_J = P_J_matrix.rank()
f = filtered_exp1
f_vector = sp.Matrix([f])
variables = sp.Matrix([x1, x2, x3, x4, x5])
grad_f1 = f_vector.jacobian(variables)
sp.pprint(grad_f1)
grad_f1_shape = grad_f1.shape
x1_value, x2_value, x3_value, x4_value, x5_value = values
P_J1 = grad_f1.subs({x1: x1_value, x2: x2_value, x3: x3_value, x4: x4_value, x5: x5_value})
P_J_matrix1 = Matrix(P_J1) 
rank_P_J1 = P_J_matrix1.rank() 
dJ = P_J_matrix1 * P_J_matrix 
def solve_for_lambda(data, c, d, lambda_value):
    B5d = pd.read_csv('../../../Data/Lorenz_System/Lorenz_System_T5_40/trainingp_data50.csv')
    traj_len = B5d.groupby('trajectory').size()
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
    A = M 
    F =A.T
    return A, K_with_I_inv, F
data = np.loadtxt('../../../Data/Lorenz_System/Lorenz_System_T5_40/trainingp_data50.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4))
solutions_combined = []
for i in range(7, 8):
    lambda_value = 10**(-i)
    A, K_with_I_inv, F = solve_for_lambda(data, c=1, d=2, lambda_value=lambda_value)
a1 = np.random.uniform(-3, 3, num_x_variables)
def solve_for_lambda(data, c, d, lambda_value):
    B5d = pd.read_csv('../../../Data/Lorenz_System/Lorenz_System_T5_40/trainingp_data50.csv')
    traj_len = B5d.groupby('trajectory').size()
    rep = int(round(traj_len.mean()))
    K = compute_kernel_matrix(data, c, d)
    I = np.eye(K.shape[0])
    K_with_I = K + lambda_value * I
    K_with_I_inv = np.linalg.inv(K_with_I) 
    M_matrix = sp.Matrix(K_with_I_inv)
    n = K.shape[0]
    m = 5
    M = sp.zeros(n, m)
    for i in range(n):
        for j in range(m):
            start_idx = j * rep
            end_idx = (j + 1) * rep
            M[i, j] = sp.Add(*M_matrix[i, start_idx:end_idx])
    M_transpose = M.transpose()
    A = M_transpose @ M
    F =A
    return A, K_with_I_inv, F
data = np.loadtxt('../../../Data/Lorenz_System/Lorenz_System_T5_40/trainingp_data50.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4))
solutions_combined = []
for i in range(7, 8):
    lambda_value = 10**(-i)
    A, K_with_I_inv, F = solve_for_lambda(data, c=1, d=2, lambda_value=lambda_value)
def solve_for_lambda(data, c, d, lambda_value):
    B5d = pd.read_csv('../../../Data/Lorenz_System/Lorenz_System_T5_40/trainingp_data50.csv')
    traj_len = B5d.groupby('trajectory').size()
    rep = int(round(traj_len.mean()))
    K = compute_kernel_matrix(data, c, d)
    I = np.eye(K.shape[0])
    K_with_I = K + lambda_value * I
    K_with_I_inv = np.linalg.inv(K_with_I)
    M_matrix = sp.Matrix(K_with_I_inv)
    n = K.shape[0]
    m = 5
    M = sp.zeros(n, m)
    for i in range(n):
        for j in range(m):
            start_idx = j * rep
            end_idx = (j + 1) * rep
            M[i, j] = sp.Add(*M_matrix[i, start_idx:end_idx])
    M_transpose = M.transpose()
    A = M_transpose @ M
    F = np.array(A.evalf()).astype(np.float64)
    return A, K_with_I_inv, F
N = 1
D = num_x_variables 
lower_bound = -2
upper_bound = 2
points = np.random.uniform(lower_bound, upper_bound, (N, D))
rp = ', '.join(f"{x:.8f}" for x in points[0])
values = [float(x) for x in rp.split(', ')]
x1_value, x2_value, x3_value, x4_value, x5_value = values
P_J = gradients_matrix.subs({x1: x1_value, x2: x2_value, x3: x3_value, x4: x4_value, x5: x5_value})
P_J_matrix = sp.Matrix(P_J)
rank_P_J = P_J_matrix.rank()
dJ = P_J_matrix1 * P_J_matrix
data = np.loadtxt('../../../Data/Lorenz_System/Lorenz_System_T5_40/trainingp_data50.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4))
solutions_combined = []
for i in range(7, 8):
    lambda_value = 10**(-i)
    A, K_with_I_inv, F = solve_for_lambda(data, c=1, d=2, lambda_value=lambda_value)
    q = np.zeros(F.shape[0])
    G = sparse.csc_matrix((0, F.shape[0]))
    h = np.zeros(0)
    a = np.array([values], dtype=np.float64)
    dJ_numeric = np.array(dJ.evalf()).astype(np.float64)
    r = np.array([dJ_numeric], dtype=np.float64)
    F_sparse = sparse.csc_matrix(F)
    A_eq = sparse.csc_matrix(np.vstack([a.reshape(1, -1), r.reshape(1, -1)]))
    b_eq = np.array([1.0, 0.0], dtype=np.float64)
    y = solve_qp(F_sparse, q, G, h, A_eq, b_eq, solver="clarabel")
symbolic_matrix = sp.Matrix(1, len(result), result)
ds_symbolic = ds * symbolic_matrix.T 
y_values = np.array(y)
y = sp.Matrix(y_values)
pr = y.T * ds_symbolic
pr_expr = pr[0] 
SCL = str(pr_expr)
with open("../../../results/Lorenz_System/Lorenz_System_T5_40/SCL.txt", "w") as file:
    file.write(str(SCL))
with open("../../../results/Lorenz_System/Lorenz_System_T5_40/SCL.txt", "r") as file:
    SCL = sp.sympify(file.read())
f = sp.lambdify((x1, x2, x3, x4, x5), SCL, "numpy")
dat = pd.read_csv('../../../Data/Lorenz_System/Lorenz_System_T5_40/holdoutp_data50.csv')
trajectories = dat['trajectory'].unique()
total_sum_squared_normalized_functional_value = 0
total_data_points = 0
num_x_variables = 5
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
print("Second CL:")
sp.pprint(SCL)
print("")
print("")
coefficients = list(SCL.as_coefficients_dict().values())
terms = list(SCL.as_coefficients_dict().keys())
filtered_terms = [term for coeff, term in zip(coefficients, terms) if abs(coeff) > 0.0001]
filtered_ex = sum(Mul(coeff, term) for coeff, term in zip(coefficients, terms) if term in filtered_terms)
print("Final Conservation Law:")
sp.pprint(filtered_ex)
print("")
print("")
exp2 = filtered_ex
div2 = exp2
simp_exp2 = simplify(div2)
filtered_exp2 = sum(term for term in simp_exp2.args if term.has(x1) or term.has(x2) or term.has(x3))
print("Candidate CL:")
sp.pprint(filtered_exp2)
print("")
print("")
exp = filtered_ex
tar = x5 * x3
tar_coef = exp.coeff(tar)
deno = tar_coef
div = exp / deno
simp_exp = simplify(div)
filtered_exp = sum(term for term in simp_exp.args if term.has(x1) or term.has(x2) or term.has(x3))
print("Simplified Candidate CL:")
sp.pprint(filtered_exp)
print("")
print("")
print(" Relative deviation:", standard_deviation)
print("")

### Sparsification


exprp1 = filtered_exp1
variables = sp.symbols('x1:%d' % (num_x_variables + 2))
polynomialp1 = sp.Poly(exprp1, *variables)
polynomialp1 = sp.Poly(exprp1, *variables)
coefficientsp1 = polynomialp1.coeffs(order='grevlex')
coeff_matrixp1 = sp.Matrix(coefficientsp1)
coeff_matrixp1
exprp2 = filtered_exp2
polynomialp2 = sp.Poly(exprp2, *variables)
coefficientsp2 = polynomialp2.coeffs(order='grevlex')
coeff_matrixp2 = sp.Matrix(coefficientsp2)
coeff_matrixp2
exprp2 = filtered_exp1
polynomialp2 = sp.Poly(exprp2, *variables)
order = 'grevlex' 
coefficientsp2 = polynomialp2.coeffs(order=order)
coeff_list = list(coefficientsp2)
terms = polynomialp2.monoms(order=order)
coefficients = polynomialp2.coeffs()
terms_with_coeffs = {sp.Mul(*[s**e for s, e in zip([*variables], term)]): coeff
                      for term, coeff in zip(terms, coefficients)}
for term in terms:
    term_expr = sp.Mul(*[s**e for s, e in zip([*variables], term)])
    coeff = terms_with_coeffs.get(term_expr, 0)
exprp1 = filtered_exp2
polynomialp1 = sp.Poly(exprp1, *variables)
order = 'grevlex' 
coefficientsp1 = polynomialp1.coeffs(order=order)
coeff_list = list(coefficientsp1)
terms = polynomialp1.monoms(order=order)
coefficients = polynomialp1.coeffs()
terms_with_coeffs = {sp.Mul(*[s**e for s, e in zip([*variables], term)]): coeff
                      for term, coeff in zip(terms, coefficients)}
for term in terms:
    term_expr = sp.Mul(*[s**e for s, e in zip([*variables], term)])
    coeff = terms_with_coeffs.get(term_expr, 0)
terms = polynomialp2.monoms(order=order)
coefficients = polynomialp2.coeffs()
terms_with_coeffs = {sp.Mul(*[s**e for s, e in zip([*variables], term)]): coeff
                      for term, coeff in zip(terms, coefficients)}
formatted_terms = [str(sp.Mul(*[s**e for s, e in zip([*variables], term)])) for term in terms]
formatted_terms_str = ', '.join(formatted_terms)
vp1 = [sp.Mul(*[s**e for s, e in zip([*variables], term)]).evalf() for term in terms]
if coeff_matrixp1.shape[0] != coeff_matrixp2.shape[0]:
    raise ValueError("The two coefficient matrices do not have the same number of rows.")
combined_matrixp = sp.Matrix.hstack(coeff_matrixp1, coeff_matrixp2)
combined_matrixp
C = combined_matrixp
a = cp.Variable(2)
Ca = C @ a
objective = cp.Minimize(cp.norm1(Ca))
constraints = [
    cp.sum(a) == 1
]
problem = cp.Problem(objective, constraints)
problem.solve()
a_optimal = a.value
C = combined_matrixp
a = np.array(a_optimal)
resultp = np.dot(C, a)
formatted_resultp = ', '.join(map(str, resultp))
v3_str = '[' + ', '.join(f'{x:.15g}' for x in resultp) + ']'
v3_list = ast.literal_eval(v3_str)
v3_sympy = [sp.Float(x) for x in v3_list]
v1 = [sp.Mul(*[s**e for s, e in zip([*variables], term)]).evalf() for term in terms]
dot_product1 = sum(sp.Mul(v1_i, v3_i) for v1_i, v3_i in zip(v1, v3_sympy))
dot_product1_simplified = sp.simplify(dot_product1)
print("First sparse CL:")
sp.pprint(dot_product1_simplified)
print("")
print("")
threshold = 0.0001
filtered_terms = []
for term in dot_product1_simplified.as_ordered_terms():
    coeff = term.as_coeff_Mul()[0] 
    if abs(coeff) > threshold:
        filtered_terms.append(term)
filtered_expression = sum(filtered_terms)
print("Simplified First sparse CL:")
sp.pprint(filtered_expression)
print("")
print("")
# exp3 = filtered_expression
# tar3 = x4 * x3
# tar_coef3 = exp3.coeff(tar3)
# deno3 = tar_coef3
# div3 = exp3 / deno3
# simp_exp3 = simplify(div3)
# filtered_exp3 = sum(term for term in simp_exp3.args if term.has(x1) or term.has(x2) or term.has(x3))
# print("Final First Sparse CL:")
# sp.pprint(filtered_exp3)
# print("")
# print("")
if coeff_matrixp1.shape[0] != coeff_matrixp2.shape[0]:
    raise ValueError("The two coefficient matrices do not have the same number of rows.")
combined_matrixp = sp.Matrix.hstack(coeff_matrixp1, coeff_matrixp2)
C = np.array(combined_matrixp).astype(np.float64)
a = np.array(a_optimal)
resultp = np.dot(C, a)
final_result = np.dot(resultp, C)
C = combined_matrixp
a = cp.Variable(2)
Ca = C @ a
objective = cp.Minimize(cp.norm1(Ca))
constraints = [
    cp.sum(a) == 1,
    final_result[1] * a[1] + final_result[0] * a[0] == 0  # New constraint
]
problem = cp.Problem(objective, constraints)
problem.solve()
a_optimal1 = a.value
C = combined_matrixp
a = np.array(a_optimal1)
resultp = np.dot(C, a)
formatted_resultp = ', '.join(map(str, resultp))
v3_str = '[' + ', '.join(f'{x:.15g}' for x in resultp) + ']'
v3_list = ast.literal_eval(v3_str)
v3_sympy = [sp.Float(x) for x in v3_list]
v1 = [sp.Mul(*[s**e for s, e in zip([*variables], term)]).evalf() for term in terms]
dot_product1 = sum(sp.Mul(v1_i, v3_i) for v1_i, v3_i in zip(v1, v3_sympy))
dot_product1_simplified = sp.simplify(dot_product1)
v3_str = '[' + ', '.join(f'{x:.15g}' for x in resultp) + ']'
v3_list = ast.literal_eval(v3_str)
v3_sympy = [sp.Float(x) for x in v3_list]
v1 = [sp.Mul(*[s**e for s, e in zip([*variables], term)]).evalf() for term in terms]
dot_product1 = sum(sp.Mul(v1_i, v3_i) for v1_i, v3_i in zip(v1, v3_sympy))
dot_product1_simplified = sp.simplify(dot_product1)
threshold = 0.0001
filtered_terms = []
for term in dot_product1_simplified.as_ordered_terms():
    coeff = term.as_coeff_Mul()[0] 
    if abs(coeff) > threshold:
        filtered_terms.append(term)
filtered_expression4 = sum(filtered_terms)
print("Second CL:")
sp.pprint(dot_product1_simplified)
print("")
print("")
print("Simplified second sparse CL:")
sp.pprint(filtered_expression4)
print("")
print("")
# exp4 = filtered_expression4
# tar4 = x3 * x5
# tar_coef4 = exp4.coeff(tar4)
# deno4 = tar_coef4
# div4 = exp4 / deno4
# simp_exp4 = simplify(div4)
# filtered_exp4 = sum(term for term in simp_exp4.args if term.has(x1) or term.has(x2) or term.has(x3))
# print("Final Second Sparse CL:")
# sp.pprint(filtered_exp4)
# print("")
# print("")


# In[ ]:




