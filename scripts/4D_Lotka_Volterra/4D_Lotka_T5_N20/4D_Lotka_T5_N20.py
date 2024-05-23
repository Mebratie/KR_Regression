#!/usr/bin/env python
# coding: utf-8

# In[9]:


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
# %matplotlib inline
dm = pd.read_csv("../../../System/4D_Lotka_Volterra/4D_Lotka_T5_N20/50.csv")
m = dm['trajectory'].nunique()
x_columns = [col for col in dm.columns if col.startswith('x')]
num_x_variables = len(x_columns)
num_y_variables = m
d = int(input("Enter degree for polynomial kernel: "))
def compute_kernel_matrix(X, c, d):
    n = X.shape[0]
    K = (c + np.dot(X, X.T)) ** d
    return K
def solve_for_lambda(data, c, d, lambda_value, m):
    B501_data = pd.read_csv('../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B501.csv')
    traj_len = B501_data.groupby('trajectory').size()
    rep = int(round(traj_len.mean()))
    K = compute_kernel_matrix(data, c, d)
    I = np.eye(K.shape[0])
    K_with_I = K + lambda_value * I
    K_with_I_inv = np.linalg.inv(K_with_I)
    x_symbols = [sp.symbols(f'x{i}') for i in range(1, num_x_variables + 1)]
    y_symbols = [sp.symbols(f'y{i}') for i in range(num_y_variables)]
    y0 = sp.symbols('y0')
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
    C = y.T
    D = C @ A @ y
    D_expanded = sp.expand(D.subs(y0, 1))
    W1 = sp.Matrix([sp.diff(D_expanded, var) for var in y])
    B = W1.subs(y0, 1)
    system_of_equations = []
    variables = [sp.symbols(f'y{i}') for i in range(1, m)]
    for i in range(len(y_pattern)-1):
        equation = sp.Eq(B[i+1], 0)
        system_of_equations.append(equation)
    solution = sp.solve(system_of_equations, variables)
    solution_with_y0 = {**{'y0': 1}, **solution}
    return {'W1': W1, 'solution': solution_with_y0}
def solve_for_lambda_A2(data_A2, c, d, lambda_value, m):
    B501_data = pd.read_csv('../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B501.csv')
    traj_len = B501_data.groupby('trajectory').size()
    rep = int(round(traj_len.mean()))
    K = compute_kernel_matrix(data_A2, c, d)
    I = np.eye(K.shape[0])
    K_with_I = K + lambda_value * I
    K_with_I_inv = np.linalg.inv(K_with_I)
    x_symbols = [sp.symbols(f'x{i}') for i in range(1, num_x_variables + 1)]
    y_symbols = [sp.symbols(f'y{i}') for i in range(num_y_variables)]
    y0 = sp.symbols('y0')
    substitutions = {old: new for old, new in zip([f'x{i}' for i in range(1, num_x_variables + 1)], x_symbols)}
    substitutions.update({old: new for old, new in zip([f'y{i}' for i in range(num_y_variables)], y_symbols)})
    K_with_I_inv_A2 = sp.Matrix(K_with_I_inv).subs(substitutions)
    y_pattern = [sp.symbols(f'y{i}') for i in range(m)]
    y_repeated = np.repeat(y_pattern, rep, axis=0)
    y = sp.Matrix([sp.symbols(f'y{i}') for i in range(m)])
    M_matrix = sp.Matrix(K_with_I_inv_A2)
    n = K.shape[0]
    M = sp.zeros(n, m)
    for i in range(n):
        for j in range(m):
            start_idx = j * rep
            end_idx = (j + 1) * rep
            M[i, j] = sp.Add(*M_matrix[i, start_idx:end_idx])
    M_transpose = M.transpose()
    A = M_transpose @ M
    C = y.T
    D = C @ A @ y
    D_expanded = sp.expand(D.subs(y0, 1))
    W2 = sp.Matrix([sp.diff(D_expanded, var) for var in y])
    B = W2.subs(y0, 1)
    system_of_equations = []
    variables = [sp.symbols(f'y{i}') for i in range(1, m)]
    for i in range(len(y_pattern)-1):
        equation = sp.Eq(B[i+1], 0)
        system_of_equations.append(equation)
    solution = sp.solve(system_of_equations, variables)
    solution_with_y0 = {**{'y0': 1}, **solution}
    return {'W2': W2, 'solution': solution_with_y0}
def solve_for_lambda_A3(data_A3, c, d, lambda_value, m):
    B501_data = pd.read_csv('../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B501.csv')
    traj_len = B501_data.groupby('trajectory').size()
    rep = int(round(traj_len.mean()))
    K = compute_kernel_matrix(data_A2, c, d)
    I = np.eye(K.shape[0])
    K_with_I = K + lambda_value * I
    K_with_I_inv = np.linalg.inv(K_with_I)
    x_symbols = [sp.symbols(f'x{i}') for i in range(1, num_x_variables + 1)]
    y_symbols = [sp.symbols(f'y{i}') for i in range(num_y_variables)]
    y0 = sp.symbols('y0')
    substitutions = {old: new for old, new in zip([f'x{i}' for i in range(1, num_x_variables + 1)], x_symbols)}
    substitutions.update({old: new for old, new in zip([f'y{i}' for i in range(num_y_variables)], y_symbols)})
    K_with_I_inv_A3 = sp.Matrix(K_with_I_inv).subs(substitutions)
    y_pattern = [sp.symbols(f'y{i}') for i in range(m)]
    y_repeated = np.repeat(y_pattern, rep, axis=0)
    y = sp.Matrix([sp.symbols(f'y{i}') for i in range(m)])
    M_matrix = sp.Matrix(K_with_I_inv_A3)
    n = K.shape[0]
    M = sp.zeros(n, m)
    for i in range(n):
        for j in range(m):
            start_idx = j * rep
            end_idx = (j + 1) * rep
            M[i, j] = sp.Add(*M_matrix[i, start_idx:end_idx])
    M_transpose = M.transpose()
    A = M_transpose @ M
    C = y.T
    D = C @ A @ y
    D_expanded = sp.expand(D.subs(y0, 1))
    W3 = sp.Matrix([sp.diff(D_expanded, var) for var in y])
    B = W3.subs(y0, 1)
    system_of_equations = []
    variables = [sp.symbols(f'y{i}') for i in range(1, m)]
    for i in range(len(y_pattern)-1):
        equation = sp.Eq(B[i+1], 0)
        system_of_equations.append(equation)
    solution = sp.solve(system_of_equations, variables)
    solution_with_y0 = {**{'y0': 1}, **solution}
    return {'W3': W3, 'solution': solution_with_y0}
def solve_for_lambda_A4(data_A4, c, d, lambda_value, m):
    B501_data = pd.read_csv('../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B501.csv')
    traj_len = B501_data.groupby('trajectory').size()
    rep = int(round(traj_len.mean()))
    K = compute_kernel_matrix(data_A2, c, d)
    I = np.eye(K.shape[0])
    K_with_I = K + lambda_value * I
    K_with_I_inv = np.linalg.inv(K_with_I)
    x_symbols = [sp.symbols(f'x{i}') for i in range(1, num_x_variables + 1)]
    y_symbols = [sp.symbols(f'y{i}') for i in range(num_y_variables)]
    y0 = sp.symbols('y0')
    substitutions = {old: new for old, new in zip([f'x{i}' for i in range(1, num_x_variables + 1)], x_symbols)}
    substitutions.update({old: new for old, new in zip([f'y{i}' for i in range(num_y_variables)], y_symbols)})
    K_with_I_inv_A4 = sp.Matrix(K_with_I_inv).subs(substitutions)
    y_pattern = [sp.symbols(f'y{i}') for i in range(m)]
    y_repeated = np.repeat(y_pattern, rep, axis=0)
    y = sp.Matrix([sp.symbols(f'y{i}') for i in range(m)])
    M_matrix = sp.Matrix(K_with_I_inv_A4)
    n = K.shape[0]
    M = sp.zeros(n, m)
    for i in range(n):
        for j in range(m):
            start_idx = j * rep
            end_idx = (j + 1) * rep
            M[i, j] = sp.Add(*M_matrix[i, start_idx:end_idx])
    M_transpose = M.transpose()
    A = M_transpose @ M
    C = y.T
    D = C @ A @ y
    D_expanded = sp.expand(D.subs(y0, 1))
    W4 = sp.Matrix([sp.diff(D_expanded, var) for var in y])
    B = W4.subs(y0, 1)
    system_of_equations = []
    variables = [sp.symbols(f'y{i}') for i in range(1, m)]
    for i in range(len(y_pattern)-1):
        equation = sp.Eq(B[i+1], 0)
        system_of_equations.append(equation)
    solution = sp.solve(system_of_equations, variables)
    solution_with_y0 = {**{'y0': 1}, **solution}
    return {'W4': W4, 'solution': solution_with_y0}
def add_matrices(mat1, mat2, mat3, mat4):
    return mat1 + mat2 + mat3 + mat4
data = np.loadtxt('../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B501.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
data_A2 = np.loadtxt('../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B502.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
data_A3 = np.loadtxt('../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B503.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
data_A4 = np.loadtxt('../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B504.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
lamda = 8 # order of magnitude for lambda value
lambda_values_A1 = [10**(-i) for i in range(lamda)]
lambda_values_A2 = [10**(-i) for i in range(lamda)]
lambda_values_A3 = [10**(-i) for i in range(lamda)]
lambda_values_A4 = [10**(-i) for i in range(lamda)]
c = 1 # Adjust the values of variable "d" according to the required degree.
c_A1 = c_A2 = c_A3 = c_A4 = c
d_A1 = d_A2 = d_A3 = d_A4 = d
solutions_A1 = [solve_for_lambda(data, c_A1, d_A1, lambda_val_A1, m) for lambda_val_A1 in lambda_values_A1]
solutions_A2 = [solve_for_lambda_A2(data_A2, c_A2, d_A2, lambda_val_A2, m) for lambda_val_A2 in lambda_values_A2]
solutions_A3 = [solve_for_lambda_A3(data_A3, c_A3, d_A3, lambda_val_A3, m) for lambda_val_A3 in lambda_values_A3]
solutions_A4 = [solve_for_lambda_A4(data_A4, c_A4, d_A4, lambda_val_A4, m) for lambda_val_A4 in lambda_values_A4]
solutions_combined = []
for i, (w1, w2, w3, w4) in enumerate(zip(solutions_A1, solutions_A2, solutions_A3, solutions_A4)):
    added_matrix = add_matrices(w1['W1'], w2['W2'], w3['W3'], w4['W4'])
    symbols_A1 = w1['W1'].free_symbols
    symbols_A2 = w2['W2'].free_symbols
    symbols_A3 = w3['W3'].free_symbols
    symbols_A4 = w4['W4'].free_symbols
    all_symbols = symbols_A1.union(symbols_A2).union(symbols_A3).union(symbols_A4)
    set_solution = {'y0': 1}
    equations = [sp.Eq(added_matrix[i], 0) for i in range(added_matrix.shape[0])]
    solutions_system = sp.solve(equations, all_symbols)
    for var, sol in solutions_system.items():
        set_solution[str(var)] = sol    
    solutions_combined.append(set_solution)
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
def process_dataset(file_path, lambda_values, y_values_dicts):
  data = pd.read_csv(file_path)
  variables = ["c"] + x_columns
  data['c'] = 1
  variables1 = ["1"] + x_columns
  all_entries = []
  f_alpha_expression_list = []
  data_np = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
  for index, row in data.iterrows():
      result = multilinear_expansion(variables, d, row)
      result1 = multilinear_expansion1(variables1, d)
      expressions = generate_inner_products(result, result1)
      all_entries.append(expressions)
  for lambda_val, y_values_dict in zip(lambda_values, y_values_dicts_list):
        c = 1.0
        K_with_I_inv, alpha_sym = solve_for_lambda(data_np, c, d, lambda_val, list(y_values_dict.values()))
        total_sum = 0
#         print(f"File: {file_path}, Lambda = 10^(-{lambda_values.index(lambda_val)}), y_values_dict: {y_values_dict}:")
        for entry_index, (entry, alpha) in enumerate(zip(all_entries, alpha_sym), 1):
            entry_sum = 0
            for term in entry:
                result = alpha * sp.sympify(term)
                entry_sum += result
                #print(result)
            total_sum += entry_sum
        f_alpha_expression_list.append(total_sum)
  return f_alpha_expression_list
lambda_values = [10**(-i) for i in range(lamda)]
y_values_dicts_list = solutions_combined
file_path_1 = "../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B501.csv"
file_path_2 = "../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B502.csv"
file_path_3 = "../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B503.csv"
file_path_4 = "../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B504.csv"
B501_data = pd.read_csv('../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B501.csv')
df = pd.read_csv("../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/B505.csv")
traj_len = B501_data.groupby('trajectory').size()
rep = int(round(traj_len.mean()))
f_alpha_expression_1 = process_dataset(file_path_1,  lambda_values, y_values_dicts_list)
f_alpha_expression_2 = process_dataset(file_path_2,  lambda_values, y_values_dicts_list)
f_alpha_expression_3 = process_dataset(file_path_3,  lambda_values, y_values_dicts_list)
f_alpha_expression_4 = process_dataset(file_path_4,  lambda_values, y_values_dicts_list)
f_vector1_0, f_vector1_1, f_vector1_2, f_vector1_3, f_vector1_4, f_vector1_5, f_vector1_6, f_vector1_7 = [], [], [], [], [], [], [], []
for index, row in df.iterrows():
    value1_0, value1_1 = eval(str(f_alpha_expression_1[0]), globals(), row.to_dict()), eval(str(f_alpha_expression_1[1]), globals(), row.to_dict())
    value1_2, value1_3 = eval(str(f_alpha_expression_1[2]), globals(), row.to_dict()), eval(str(f_alpha_expression_1[3]), globals(), row.to_dict())
    value1_4, value1_5 = eval(str(f_alpha_expression_1[4]), globals(), row.to_dict()), eval(str(f_alpha_expression_1[5]), globals(), row.to_dict())
    value1_6, value1_7 = eval(str(f_alpha_expression_1[6]), globals(), row.to_dict()), eval(str(f_alpha_expression_1[7]), globals(), row.to_dict())
    f_vector1_0.append(value1_0),f_vector1_1.append(value1_1),f_vector1_2.append(value1_2),f_vector1_3.append(value1_3), f_vector1_4.append(value1_4), f_vector1_5.append(value1_5), f_vector1_6.append(value1_6), f_vector1_7.append(value1_7)
f_vector2_0, f_vector2_1, f_vector2_2, f_vector2_3, f_vector2_4, f_vector2_5, f_vector2_6, f_vector2_7 = [], [], [], [], [], [], [], []
for index, row in df.iterrows():
    value2_0, value2_1 = eval(str(f_alpha_expression_2[0]), globals(), row.to_dict()), eval(str(f_alpha_expression_2[1]), globals(), row.to_dict())
    value2_2, value2_3 = eval(str(f_alpha_expression_2[2]), globals(), row.to_dict()), eval(str(f_alpha_expression_2[3]), globals(), row.to_dict())
    value2_4, value2_5 = eval(str(f_alpha_expression_2[4]), globals(), row.to_dict()), eval(str(f_alpha_expression_2[5]), globals(), row.to_dict())
    value2_6, value2_7 = eval(str(f_alpha_expression_2[6]), globals(), row.to_dict()), eval(str(f_alpha_expression_2[7]), globals(), row.to_dict())
    f_vector2_0.append(value2_0),f_vector2_1.append(value2_1),f_vector2_2.append(value2_2),f_vector2_3.append(value2_3),f_vector2_4.append(value2_4),f_vector2_5.append(value2_5),f_vector2_6.append(value2_6),f_vector2_7.append(value2_7)
f_vector3_0, f_vector3_1, f_vector3_2, f_vector3_3, f_vector3_4, f_vector3_5, f_vector3_6, f_vector3_7 = [], [], [], [], [], [], [], []
for index, row in df.iterrows():
    value3_0, value3_1 = eval(str(f_alpha_expression_3[0]), globals(), row.to_dict()), eval(str(f_alpha_expression_3[1]), globals(), row.to_dict())
    value3_2, value3_3 = eval(str(f_alpha_expression_3[2]), globals(), row.to_dict()), eval(str(f_alpha_expression_3[3]), globals(), row.to_dict())
    value3_4, value3_5 = eval(str(f_alpha_expression_3[4]), globals(), row.to_dict()), eval(str(f_alpha_expression_3[5]), globals(), row.to_dict())
    value3_6, value3_7 = eval(str(f_alpha_expression_3[6]), globals(), row.to_dict()), eval(str(f_alpha_expression_3[7]), globals(), row.to_dict())
    f_vector3_0.append(value3_0),f_vector3_1.append(value3_1),f_vector3_2.append(value3_2),f_vector3_3.append(value3_3),f_vector3_4.append(value3_4),f_vector3_5.append(value3_5),f_vector3_6.append(value3_6),f_vector3_7.append(value3_7),
f_vector4_0, f_vector4_1, f_vector4_2, f_vector4_3, f_vector4_4, f_vector4_5, f_vector4_6, f_vector4_7 = [], [], [], [], [], [], [], []
for index, row in df.iterrows():
    value4_0, value4_1 = eval(str(f_alpha_expression_4[0]), globals(), row.to_dict()), eval(str(f_alpha_expression_4[1]), globals(), row.to_dict())
    value4_2, value4_3 = eval(str(f_alpha_expression_4[2]), globals(), row.to_dict()), eval(str(f_alpha_expression_4[3]), globals(), row.to_dict())
    value4_4, value4_5 = eval(str(f_alpha_expression_4[4]), globals(), row.to_dict()), eval(str(f_alpha_expression_4[5]), globals(), row.to_dict())
    value4_6, value4_7 = eval(str(f_alpha_expression_4[6]), globals(), row.to_dict()), eval(str(f_alpha_expression_4[7]), globals(), row.to_dict())
    f_vector4_0.append(value4_0),f_vector4_1.append(value4_1),f_vector4_2.append(value4_2),f_vector4_3.append(value4_3),f_vector4_4.append(value4_4),f_vector4_5.append(value4_5),f_vector4_6.append(value4_6),f_vector4_7.append(value4_7)
h1, h2, h3, h4 = list(y_values_dicts_list[0].values()), list(y_values_dicts_list[1].values()), list(y_values_dicts_list[2].values()), list(y_values_dicts_list[3].values())
h5, h6, h7, h8 = list(y_values_dicts_list[4].values()), list(y_values_dicts_list[5].values()), list(y_values_dicts_list[6].values()), list(y_values_dicts_list[7].values())
y_B501_0, y_B501_1, y_B501_2, y_B501_3, y_B501_4 = np.repeat(h1, rep), np.repeat(h2, rep), np.repeat(h3, rep), np.repeat(h4, rep), np.repeat(h5, rep)
y_B501_5, y_B501_6, y_B501_7, y_B502_0, y_B502_1 = np.repeat(h6, rep), np.repeat(h7, rep), np.repeat(h8, rep), np.repeat(h1, rep), np.repeat(h2, rep)
y_B502_2, y_B502_3, y_B502_4, y_B502_5, y_B502_6 = np.repeat(h3, rep), np.repeat(h4, rep), np.repeat(h5, rep), np.repeat(h6, rep), np.repeat(h7, rep)
y_B502_7, y_B503_0, y_B503_1,y_B503_2, y_B503_3 = np.repeat(h8, rep), np.repeat(h1, rep), np.repeat(h2, rep), np.repeat(h3, rep), np.repeat(h4, rep)
y_B503_4,y_B503_5, y_B503_6, y_B503_7, y_B504_0 = np.repeat(h5, rep), np.repeat(h6, rep), np.repeat(h7, rep), np.repeat(h8, rep), np.repeat(h1, rep)
y_B504_1, y_B504_2, y_B504_3, y_B504_4, y_B504_5 = np.repeat(h2, rep), np.repeat(h3, rep), np.repeat(h4, rep), np.repeat(h5, rep), np.repeat(h6, rep)
y_B504_6, y_B504_7 = np.repeat(h7, rep), np.repeat(h8, rep)
RMSE_1_0, RMSE_1_1 = np.sqrt(mean_squared_error(y_B501_0, f_vector1_0)), np.sqrt(mean_squared_error(y_B501_1, f_vector1_1))
RMSE_1_2, RMSE_1_3 = np.sqrt(mean_squared_error(y_B501_2, f_vector1_2)), np.sqrt(mean_squared_error(y_B501_3, f_vector1_3))
RMSE_1_4, RMSE_1_5 = np.sqrt(mean_squared_error(y_B501_4, f_vector1_4)), np.sqrt(mean_squared_error(y_B501_5, f_vector1_5))
RMSE_1_6, RMSE_1_7 = np.sqrt(mean_squared_error(y_B501_6, f_vector1_6)), np.sqrt(mean_squared_error(y_B501_7, f_vector1_7))
RMSE_2_0, RMSE_2_1 = np.sqrt(mean_squared_error(y_B502_0, f_vector2_0)), np.sqrt(mean_squared_error(y_B502_1, f_vector2_1))
RMSE_2_2, RMSE_2_3 = np.sqrt(mean_squared_error(y_B502_2, f_vector2_2)), np.sqrt(mean_squared_error(y_B502_3, f_vector2_3))
RMSE_2_4, RMSE_2_5 = np.sqrt(mean_squared_error(y_B502_4, f_vector2_4)), np.sqrt(mean_squared_error(y_B502_5, f_vector2_5))
RMSE_2_6, RMSE_2_7 = np.sqrt(mean_squared_error(y_B502_6, f_vector2_6)), np.sqrt(mean_squared_error(y_B502_7, f_vector2_7))
RMSE_3_0, RMSE_3_1 = np.sqrt(mean_squared_error(y_B503_0, f_vector3_0)), np.sqrt(mean_squared_error(y_B503_1, f_vector3_1))
RMSE_3_2, RMSE_3_3 = np.sqrt(mean_squared_error(y_B503_2, f_vector3_2)), np.sqrt(mean_squared_error(y_B503_3, f_vector3_3))
RMSE_3_4, RMSE_3_5 = np.sqrt(mean_squared_error(y_B503_4, f_vector3_4)), np.sqrt(mean_squared_error(y_B503_5, f_vector3_5))
RMSE_3_6, RMSE_3_7 = np.sqrt(mean_squared_error(y_B503_6, f_vector3_6)), np.sqrt(mean_squared_error(y_B503_7, f_vector3_7))
RMSE_4_0, RMSE_4_1 = np.sqrt(mean_squared_error(y_B504_0, f_vector4_0)), np.sqrt(mean_squared_error(y_B504_1, f_vector4_1))
RMSE_4_2, RMSE_4_3 = np.sqrt(mean_squared_error(y_B504_2, f_vector4_2)), np.sqrt(mean_squared_error(y_B504_3, f_vector4_3))
RMSE_4_4, RMSE_4_5 = np.sqrt(mean_squared_error(y_B504_4, f_vector4_4)), np.sqrt(mean_squared_error(y_B504_5, f_vector4_5))
RMSE_4_6, RMSE_4_7 = np.sqrt(mean_squared_error(y_B504_6, f_vector4_6)), np.sqrt(mean_squared_error(y_B504_7, f_vector4_7))
rmse_values = {
    "B501": [RMSE_1_0, RMSE_1_1, RMSE_1_2, RMSE_1_3, RMSE_1_4, RMSE_1_5, RMSE_1_6, RMSE_1_7],
    "B502": [RMSE_2_0, RMSE_2_1, RMSE_2_2, RMSE_2_3, RMSE_2_4, RMSE_2_5, RMSE_2_6, RMSE_2_7],
    "B503": [RMSE_3_0, RMSE_3_1, RMSE_3_2, RMSE_3_3, RMSE_3_4, RMSE_3_5, RMSE_3_6, RMSE_3_7],
    "B504": [RMSE_4_0, RMSE_4_1, RMSE_4_2, RMSE_4_3, RMSE_4_4, RMSE_4_5, RMSE_4_6, RMSE_4_7]
}
for file, rmse_list in rmse_values.items():
    min_rmse = min(rmse_list)
    min_index = rmse_list.index(min_rmse)
rmse_values = [RMSE_1_0, RMSE_1_1, RMSE_1_2, RMSE_1_3, RMSE_1_4, RMSE_1_5, RMSE_1_6, RMSE_1_7,
               RMSE_2_0, RMSE_2_1, RMSE_2_2, RMSE_2_3, RMSE_2_4, RMSE_2_5, RMSE_2_6, RMSE_2_7,
               RMSE_3_0, RMSE_3_1, RMSE_3_2, RMSE_3_3, RMSE_3_4, RMSE_3_5, RMSE_3_6, RMSE_3_7,
               RMSE_4_0, RMSE_4_1, RMSE_4_2, RMSE_4_3, RMSE_4_4, RMSE_4_5, RMSE_4_6, RMSE_4_7]
min_rmse = min(rmse_values)
min_index = rmse_values.index(min_rmse)
file_index = min_index // 8 + 1
sub_index = min_index % 8
h_values = [h1, h2, h3, h4, h5, h6, h7, h8]
lambda_values = [10**(-i) for i in range(lamda)]
h_value = h_values[sub_index]
df2 = pd.read_csv('../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/trainingp_data50.csv')
df2['trajectory'] = df2['trajectory'].replace({i: h_value[i-1] for i in range(1, m+1)})
X_train = df2.iloc[:, :-1]
y_train = df2.iloc[:, -1]
X_train.to_csv('../../../results/4D_Lotka_Volterra/4D_Lotka_T5_N20/X_train.csv', index=False)
y_train = y_train.astype(float)
X_train = X_train.astype(float)
def polynomial_kernel(X, Y, degree=d):
    return (1 + np.dot(X, Y.T)) ** degree
param_grid = {'alpha': [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
kr_model = KernelRidge(kernel=polynomial_kernel)
grid_search = GridSearchCV(kr_model, param_grid, cv=cv, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best RMSE:", -grid_search.best_score_)
print("")
print("")
# grid_search.best_estimator_
# y_pred = grid_search.predict(X_test)
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
x1, x2, x3, x4, x5, x6, x7, x8 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8')
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
data = pd.read_csv('../../../results/4D_Lotka_Volterra/4D_Lotka_T5_N20/X_train.csv')
variables = ['c'] + [f'x{i}' for i in range(1, num_x_variables + 1)]
data['c'] = 1
variables1 = ['1'] + [f'x{i}' for i in range(1, num_x_variables + 1)]
all_entries = []
for index, row in data.iterrows():
    result = multilinear_expansion(variables, d, row)
    result1 = multilinear_expansion1(variables1, d)
    expressions = generate_inner_products(result, result1)
    all_entries.append(expressions)
total_sum = 0
for entry_index, (entry, alpha) in enumerate(zip(all_entries, eta), 1):
    entry_sum = 0
    for term in entry:
        result = alpha * sp.sympify(term)
        entry_sum += result
    total_sum += entry_sum
coefficients = list(total_sum.as_coefficients_dict().values())
terms = list(total_sum.as_coefficients_dict().keys())
filtered_terms = [term for coeff, term in zip(coefficients, terms) if abs(coeff) > 0.001]
filtered_expression = sum(sp.Mul(coeff, term) for coeff, term in zip(coefficients, terms) if term in filtered_terms)
print("Total sum:")
sp.pprint(total_sum)
with open("../../../results/4D_Lotka_Volterra/4D_Lotka_T5_N20/total_sum.txt", "w") as file:
    file.write(str(total_sum))
print("Final Candidate CL:")
sp.pprint(filtered_expression)
print("")
print("")
coef = list(total_sum.as_coefficients_dict().values())
terms = list(total_sum.as_coefficients_dict().keys())
filtered_terms = [term for coeff, term in zip(coef, terms) if abs(coeff) > 0.0001] # remove all terms which is less that 0.0001
filtered = sum(sp.Mul(coeff, term) for coeff, term in zip(coef, terms) if term in filtered_terms)
print("Final Candidate CL:")
sp.pprint(filtered)
print("")
print("")
with open("../../../results/4D_Lotka_Volterra/4D_Lotka_T5_N20/total_sum.txt", "w") as file:
    file.write(str(total_sum))
with open("../../../results/4D_Lotka_Volterra/4D_Lotka_T5_N20/total_sum.txt", "r") as file:
    total_sum = sp.sympify(file.read())
df3 = pd.read_csv('../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/holdoutp_data50.csv')
traj_len = df3.groupby('trajectory').size()
rep1 = int(round(traj_len.mean()))
expression = sp.lambdify((x1, x2, x3, x4, x5, x6, x7, x8), total_sum, "numpy")
df3['lamhold'] = expression(df3['x1'], df3['x2'], df3['x3'], df3['x4'], df3['x5'], df3['x6'], df3['x7'], df3['x8'])
da = {'y{}'.format(i): h_value[i] for i in range(len(h_value))}
df3['Coluh(lamhold)'] = [da[f'y{i}'] for i in range(m) for _ in range(rep1)]
columns_to_compare = [('lamhold', 'Coluh(lamhold)')]
for col1, col2 in columns_to_compare:
    rmse = np.sqrt(mean_squared_error(df3[col1], df3[col2]))
    print(f'Generalisation Error (RMSE): {rmse}')
    print("")
with open("../../../results/4D_Lotka_Volterra/4D_Lotka_T5_N20/total_sum.txt", "r") as file:
    total_sum = sp.sympify(file.read())
f = sp.lambdify((x1, x2, x3, x4, x5, x6, x7, x8), total_sum, "numpy")
dat = pd.read_csv('../../../System/4D_Lotka_Volterra/4D_Lotka_T5_N20/50.csv')
trajectories = dat['trajectory'].unique()
total_sum_squared_normalized_functional_value = 0
total_data_points = 0
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
df['x1'] = pd.to_numeric(df['x1'], errors='coerce')
df['x2'] = pd.to_numeric(df['x2'], errors='coerce')
df['x3'] = pd.to_numeric(df['x3'], errors='coerce')
df['x4'] = pd.to_numeric(df['x4'], errors='coerce')
df['x5'] = pd.to_numeric(df['x5'], errors='coerce')
df['x6'] = pd.to_numeric(df['x6'], errors='coerce')
df['x7'] = pd.to_numeric(df['x7'], errors='coerce')
df['x8'] = pd.to_numeric(df['x8'], errors='coerce')
f1_func = lambdify((x1, x2, x3, x4, x5, x6, x7, x8), filtered, "numpy")
def f2(x1, x2, x3, x4, x5, x6, x7, x8):
    return x1+x2+x3+x4-x5-x6-x7-x8 # Exact Conservation law
dat['f1'] = f1_func(dat['x1'], dat['x2'], dat['x3'], dat['x4'], dat['x5'], dat['x6'], dat['x7'], dat['x8'])
dat['f2'] = dat.apply(lambda row: f2(row['x1'], row['x2'], row['x3'], row['x4'], row['x5'], row['x6'], row['x7'], row['x8']), axis=1)
corr, _ = pearsonr(dat['f1'], dat['f2'])
# print(f'Correlation coefficient: {corr}')
plt.scatter(dat['f1'], dat['f2'], s=100, c=dat['trajectory'], cmap='viridis')
plt.xlabel('Learned', fontsize=18)
plt.ylabel('Exact', fontsize=18)
plt.xticks([])
plt.yticks([])
save_path = '../../../results/4D_Lotka_Volterra/4D_Lotka_T5_N20/Coerrelation_plot.pdf'
plt.savefig(save_path)
plt.close()
input_directory = r'../../../System/4D_Lotka_Volterra/4D_Lotka_T5_N20'
output_directory = r'../../../Data/4D_Lotka_Volterra/4D_Lotka_T5_N20/d'
df14 = pd.read_csv(os.path.join(input_directory, '50.csv')) # 50.csv, saved names of the data
Br = pd.read_csv('../../../System/4D_Lotka_Volterra/4D_Lotka_T5_N20/50.csv')
tr = Br.groupby('trajectory').size()
re1 = int(round(tr.mean()))
rows_per_file = re1 
num_files = len(df14) // rows_per_file
data_chunks = [df14.iloc[i * rows_per_file:(i + 1) * rows_per_file] for i in range(num_files)]
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
for i, chunk in enumerate(data_chunks):
    chunk.to_csv(os.path.join(output_directory, f'q{i + 1}.csv'), index=False)
expression = total_sum
with open('../../../System/4D_Lotka_Volterra/4D_Lotka_T5_N20/50.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader) 
    current_trajectory = None
    expression_values = []
    for row in csv_reader:
        x1_val, x2_val, x3_val, x4_val, x5_val, x6_val, x7_val, x8_val, trajectory = map(float, row)
        if trajectory != current_trajectory:
            current_trajectory = trajectory
            initial_data = [x1_val, x2_val, x3_val, x4_val, x5_val, x6_val, x7_val, x8_val]
            expression_value = expression.subs({x1: x1_val, x2: x2_val, x3: x3_val, x4: x4_val, x5: x5_val, x6: x6_val, x7: x7_val, x8: x8_val})
            expression_values.append(expression_value)
compute_function = sp.lambdify((x1, x2, x3, x4, x5, x6, x7, x8), total_sum, 'numpy')
for i in range(1, m+1):
    input_file = os.path.join(output_directory, f'q{i}.csv')
    output_file = os.path.join(output_directory, f'r{i}.csv')
    data = pd.read_csv(input_file)
    data['computed_value'] = compute_function(data['x1'], data['x2'], data['x3'], data['x4'], data['x5'], data['x6'], data['x7'], data['x8'])
    data.to_csv(output_file, index=False)
subtraction_values = expression_values
for i in range(1, m+1): # 5 trajectory
    r_file = os.path.join(output_directory, f'r{i}.csv')
    n_file = os.path.join(output_directory, f'n{i}.csv')
    subtraction_value = subtraction_values[i - 1]
    data = pd.read_csv(r_file)
    data['adjusted_value'] = data['computed_value'] - subtraction_value
    data.to_csv(n_file, columns=['adjusted_value'], index=False)
all_data = pd.DataFrame()
subtraction_values = expression_values
for i in range(1, m+1): # 5 trajectory
    r_file = os.path.join(output_directory, f'r{i}.csv')
    n_file = os.path.join(output_directory, f'n{i}.csv')
    subtraction_value = subtraction_values[i - 1]
    data = pd.read_csv(r_file)
    data['adjusted_value'] = (data['computed_value'] - subtraction_value)/subtraction_value
    data.to_csv(n_file, columns=['adjusted_value'], index=False)
    all_data[f'n{i}'] = data['adjusted_value']
plt.figure(figsize=(10, 6))
for column in all_data.columns:
    plt.plot(all_data[column], label=column)
plt.xlabel('N', fontsize=16)
plt.ylabel('Relative Variation', fontsize=16)
save_path = '../../../results/4D_Lotka_Volterra/4D_Lotka_T5_N20/Relative_Variation.png'
plt.savefig(save_path)
plt.close()


# In[ ]:




