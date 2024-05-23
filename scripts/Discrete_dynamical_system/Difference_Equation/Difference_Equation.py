#!/usr/bin/env python
# coding: utf-8


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
import numpy as np
import sympy as sp
from sympy import symbols, simplify, Mul, expand
dm = pd.read_csv('../../../System/Discrete_dynamical_system/Difference_Equation/50.csv')
traj_len = dm.groupby('trajectory').size()
rep = int(round(traj_len.mean()))
data = np.loadtxt('../../../System/Discrete_dynamical_system/Difference_Equation/50.csv', delimiter=',', skiprows=1, usecols=(0, 1,2))
m = dm['trajectory'].nunique()
x_columns = [col for col in dm.columns if col.startswith('x')]
num_x_variables = len(x_columns)
num_y_variables = m
d = int(input("Enter degree for polynomial kernel: "))
x1, x, x3, r = data[:, 0], data[:, 1], data[:, 2], 0.00001
def compute_kernel_matrix(X, c, d):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = (c + (X[i, 0]*X[j, 0]) + (X[i, 1]*X[j, 1])+ (X[i, 2]*X[j, 2])) ** d
    return K
c = 1.0  
K = compute_kernel_matrix(data, c, d)
I = np.eye(K.shape[0])
K_with_I = K + r*I
K_with_I_inv = np.linalg.inv(K_with_I)
x_1, x_2, x_3, y0, y1, y2 = sp.symbols('x_1 x_2 x_3 y0 y1 y2')
y_pattern = [y0, y1, y2]
y_repeated = np.repeat(y_pattern, rep, axis=0)
y = sp.Matrix([y0, y1, y2])
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
D_expanded = sp.expand(D)
W = sp.Matrix([sp.diff(D_expanded, var) for var in y])
B = W.subs(y0, 1)
system_of_equations = []
variables = [y1, y2]
for i in range(len(y_pattern)-1):
    equation = sp.Eq(B[i+1], 0)
    system_of_equations.append(equation)
solution = sp.solve(system_of_equations, variables)
x1, x2, x3 = data[:, 0], data[:, 1], data[:, 2]
def compute_kernel_matrix(X, c, d):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = (c + (X[i, 0]*X[j, 0]) + (X[i, 1]*X[j, 1])+ (X[i, 2]*X[j, 2])) ** d
    return K
c = 1.0  
K = compute_kernel_matrix(data, c, d)
I = np.eye(K.shape[0])
K_with_I = K + 0.00001*I
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
D_expanded = sp.expand(D)
W = sp.Matrix([sp.diff(D_expanded, var) for var in y])
B = W.subs(y0, 1)
system_of_equations = []
variables = [y1, y2]
for i in range(len(y_pattern)-1):
    equation = sp.Eq(B[i+1], 0)
    system_of_equations.append(equation)
solution = sp.solve(system_of_equations, variables)
solution_list = [1] + [solution[var] for var in variables]
y_values = solution_list
y_repeated = np.repeat(y_values, len(data) // len(y_values))
alpha_sym = K_with_I_inv @ y_repeated
x_q1_sym, x_q2_sym, x_q3_sym = sp.symbols('x1 x2 x3')
f_alpha = 0
for i in range(len(alpha_sym)):
    f_alpha += alpha_sym[i] * (c + (x1[i] * x_q1_sym) + (x2[i] * x_q2_sym)+ (x3[i] * x_q3_sym)) ** d
f_alpha_expanded = sp.expand(f_alpha)
f_alpha_collected = sp.collect(f_alpha_expanded, (x_q1_sym, x_q2_sym, x_q3_sym))
print("Candidate Conservation Law:")
sp.pprint(f_alpha_collected)
print("")
print("")
expanded_result = sp.expand(f_alpha_collected)
coefficients = list(expanded_result.as_coefficients_dict().values())
terms = list(expanded_result.as_coefficients_dict().keys())
filtered_terms = [term for coeff, term in zip(coefficients, terms) if abs(coeff) > 0.1]
filtered_expression = sum(sp.Mul(coeff, term) for coeff, term in zip(coefficients, terms) if term in filtered_terms)
expression = filtered_expression
expanded_expr = expand(expression)
x1, x2, x3 = sp.symbols('x1 x2 x3')
coefficient = expanded_expr.coeff(x1*x2*x3)
divided_expression = expression / coefficient
simplified_expression = simplify(divided_expression)
filtered_exp = sum(term for term in simplified_expression.args if term.has(x1) or term.has(x2) or term.has(x3))
print("Final Candidate CL:")
sp.pprint(expression)
print("")
print("")
print("Simplified Candidate CL:")
sp.pprint(filtered_exp)
print("")
print("")


# In[ ]:



