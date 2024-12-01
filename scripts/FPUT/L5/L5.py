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
data = np.genfromtxt('L5.csv', delimiter=',', names=True)
training_data = []
holdout_data = []
for r in range(1, 6):
    trajectory_subset = data[data['trajectory'] == r]
    train_set, holdout_set = train_test_split(trajectory_subset, test_size=0.2, random_state=42)
    training_data.extend(train_set)
    holdout_data.extend(holdout_set)
with open('trainingp_data.csv', 'w', newline='') as trainfile:
    writer = csv.writer(trainfile)
    writer.writerow(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'trajectory'])
    for row in training_data:
        writer.writerow([row['x1'], row['x2'], row['x3'], row['x4'], row['x5'], row['x6'], row['x7'], row['x8'], row['x9'], row['x10'], row['x11'], row['x12'], row['trajectory']])
with open('holdoutp_data.csv', 'w', newline='') as holdfile:
    writer = csv.writer(holdfile)
    writer.writerow(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'trajectory'])
    for row in holdout_data:
        writer.writerow([row['x1'], row['x2'], row['x3'], row['x4'], row['x5'], row['x6'], row['x7'], row['x8'], row['x9'], row['x10'], row['x11'], row['x12'], row['trajectory']])
trajectories = {1: [], 2: [], 3: [], 4: [], 5: []}
with open('trainingp_data.csv', 'r') as trainfile:
    reader = csv.DictReader(trainfile)
    for row in reader:
        x1 = float(row['x1'])
        x2 = float(row['x2'])
        x3 = float(row['x3'])
        x4 = float(row['x4'])
        x5 = float(row['x5'])
        x6 = float(row['x6'])
        x7 = float(row['x7'])
        x8 = float(row['x8'])
        x9 = float(row['x9'])
        x10 = float(row['x10'])
        x11 = float(row['x11'])
        x12 = float(row['x12'])
        trajectory = float(row['trajectory'])
        trajectories[trajectory].append({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6, 'x7':x7, 'x8': x8, 'x9': x9, 'x10': x10, 'x11': x11, 'x12': x12, 'trajectory': trajectory})
for traj_points in trajectories.values():
    random.shuffle(traj_points)
num_points_per_file = len(trajectories[1]) // 5
for i in range(5):
    output_filename = f'B{i+1}.csv'
    with open(output_filename, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'trajectory'])
        for trajectory in range(1, 6):
            points = trajectories[trajectory][i * num_points_per_file: (i + 1) * num_points_per_file]
            for point in points:
                writer.writerow([point['x1'], point['x2'], point['x3'], point['x4'], point['x5'], point['x6'], point['x7'], point['x8'], point['x9'], point['x10'], point['x11'], point['x12'], point['trajectory']])
def compute_kernel_matrix(X, c, d):
    n = X.shape[0]
    K = (c + np.dot(X, X.T)) ** d
    return K
def solve_for_lambda(data, c, d, lambda_value):
    K = compute_kernel_matrix(data, c, d)
    I = np.eye(K.shape[0])
    K_with_I = K + lambda_value * I
    K_with_I_inv = np.linalg.inv(K_with_I)
    y0, y1, y2, y3, y4 = sp.symbols('y0 y1 y2 y3 y4')
    y = sp.Matrix([y0, y1, y2, y3, y4])  
    M_matrix = sp.Matrix(K_with_I_inv)
    n = K.shape[0]
    m = 5
    M = sp.zeros(n, m)
    for i in range(n):
        for j in range(m):
            start_idx = j * 16
            end_idx = (j + 1) * 16
            M[i, j] = sp.Add(*M_matrix[i, start_idx:end_idx])
    M_transpose = M.transpose()
    A = M_transpose @ M 
    A_np = np.array(A).astype(np.float64)
    return A_np
data = np.loadtxt('B1.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
data2 = np.loadtxt('B2.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
data3 = np.loadtxt('B3.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
data4 = np.loadtxt('B4.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
solutions_combined = []
for i in range(0, 8):
    lambda_value = 10**(-i)
    A1_np = solve_for_lambda(data, c=1, d=4, lambda_value=lambda_value)
    A2_np = solve_for_lambda(data2, c=1, d=4, lambda_value=lambda_value)
    A3_np = solve_for_lambda(data3, c=1, d=4, lambda_value=lambda_value)
    A4_np = solve_for_lambda(data4, c=1, d=4, lambda_value=lambda_value)
    As_p = (A1_np + A2_np + A3_np + A4_np)
    P = np.array(As_p)
    q = np.zeros(5)
    a = np.random.uniform(-4, 4, 5)
    G = np.zeros((0, 5))
    h = np.zeros(0)
    A = a.reshape(1, -1)
    b = np.array([1.0])
    y = solve_qp(P, q, G, h, A, b, solver="clarabel")
    y_opt = y.flatten() if isinstance(y, np.ndarray) else np.array(y).flatten()
    y_names = ['y0', 'y1', 'y2', 'y3', 'y4']
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
def generate_f_alpha_expression(alpha_sym, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x_q1_sym, x_q2_sym, x_q3_sym, x_q4_sym, x_q5_sym, x_q6_sym, x_q7_sym, x_q8_sym, x_q9_sym, x_q10_sym, x_q11_sym, x_q12_sym, c, d):
    f_alpha = 0
    for i in range(len(alpha_sym)):
        f_alpha += alpha_sym[i] * (c + (x1[i] * x_q1_sym) + (x2[i] * x_q2_sym) + (x3[i] * x_q3_sym) + (x4[i] * x_q4_sym) + (x5[i] * x_q5_sym) + (x6[i] * x_q6_sym) + (x7[i] * x_q7_sym) + (x8[i] * x_q8_sym) + (x9[i] * x_q9_sym) + (x10[i] * x_q10_sym) + (x11[i] * x_q11_sym) + (x12[i] * x_q12_sym)) ** d
    f_alpha_expanded = sp.expand(f_alpha)
    f_alpha_collected = sp.collect(f_alpha_expanded, (x_q1_sym, x_q2_sym, x_q3_sym, x_q4_sym, x_q5_sym, x_q6_sym, x_q7_sym, x_q8_sym, x_q9_sym, x_q10_sym, x_q11_sym, x_q12_sym))
    return f_alpha_collected
def process_dataset(file_path, c, d, lambda_values, y_values_dicts):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6], data[:, 7], data[:, 8], data[:, 9], data[:, 10], data[:,11]
    K_with_I_inv_list = []
    alpha_sym_list = []
    f_alpha_expression_list = []
    for lambda_val, (_, y_values_dict) in zip(lambda_values, y_values_dicts):
        K_with_I_inv, alpha_sym = solve_for_lambda(data, c, d, lambda_val, list(y_values_dict.values()))
        x_q1_sym, x_q2_sym, x_q3_sym, x_q4_sym, x_q5_sym, x_q6_sym, x_q7_sym, x_q8_sym, x_q9_sym, x_q10_sym, x_q11_sym, x_q12_sym = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12')
        K_with_I_inv_list.append(K_with_I_inv)
        alpha_sym_list.append(alpha_sym)
        f_alpha_expression = generate_f_alpha_expression(alpha_sym, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x_q1_sym, x_q2_sym, x_q3_sym, x_q4_sym, x_q5_sym, x_q6_sym, x_q7_sym, x_q8_sym, x_q9_sym, x_q10_sym, x_q11_sym, x_q12_sym, c, d)
        f_alpha_expression_list.append(f_alpha_expression)
    
    return K_with_I_inv_list, alpha_sym_list, f_alpha_expression_list
lamda = 8
c, d = 1, 4
lambda_values = [10**(-i) for i in range(lamda)]
y_values_dicts_list = solutions_combined
file_path_1 = "B1.csv"
file_path_2 = "B2.csv"
file_path_3 = "B3.csv"
file_path_4 = "B4.csv"
file_path_5 = "B5.csv"
B501_data = pd.read_csv('B1.csv')
dr1 = pd.read_csv("B5.csv")
traj_len = B501_data.groupby('trajectory').size()
rep4 = int(round(traj_len.mean()))
K_with_I_inv_1, alpha_sym_1, f_alpha_expression_1 = process_dataset(file_path_1, c, d, lambda_values, y_values_dicts_list)
K_with_I_inv_2, alpha_sym_2, f_alpha_expression_2 = process_dataset(file_path_2, c, d, lambda_values, y_values_dicts_list)
K_with_I_inv_3, alpha_sym_3, f_alpha_expression_3 = process_dataset(file_path_3, c, d, lambda_values, y_values_dicts_list)
K_with_I_inv_4, alpha_sym_4, f_alpha_expression_4 = process_dataset(file_path_4, c, d, lambda_values, y_values_dicts_list)
K_with_I_inv_5, alpha_sym_5, f_alpha_expression_5 = process_dataset(file_path_5, c, d, lambda_values, y_values_dicts_list)
f_vector1_0, f_vector1_1, f_vector1_2, f_vector1_3, f_vector1_4, f_vector1_5, f_vector1_6, f_vector1_7 = [], [], [], [], [], [], [], []
for index, row in dr1.iterrows():
    value1_0, value1_1 = eval(str(f_alpha_expression_1[0]), globals(), row.to_dict()), eval(str(f_alpha_expression_1[1]), globals(), row.to_dict())
    value1_2, value1_3 = eval(str(f_alpha_expression_1[2]), globals(), row.to_dict()), eval(str(f_alpha_expression_1[3]), globals(), row.to_dict())
    value1_4, value1_5 = eval(str(f_alpha_expression_1[4]), globals(), row.to_dict()), eval(str(f_alpha_expression_1[5]), globals(), row.to_dict())
    value1_6, value1_7 = eval(str(f_alpha_expression_1[6]), globals(), row.to_dict()), eval(str(f_alpha_expression_1[7]), globals(), row.to_dict())
    f_vector1_0.append(value1_0),f_vector1_1.append(value1_1),f_vector1_2.append(value1_2),f_vector1_3.append(value1_3), f_vector1_4.append(value1_4), f_vector1_5.append(value1_5), f_vector1_6.append(value1_6), f_vector1_7.append(value1_7)
f_vector2_0, f_vector2_1, f_vector2_2, f_vector2_3, f_vector2_4, f_vector2_5, f_vector2_6, f_vector2_7 = [], [], [], [], [], [], [], []
for index, row in dr1.iterrows():
    value2_0, value2_1 = eval(str(f_alpha_expression_2[0]), globals(), row.to_dict()), eval(str(f_alpha_expression_2[1]), globals(), row.to_dict())
    value2_2, value2_3 = eval(str(f_alpha_expression_2[2]), globals(), row.to_dict()), eval(str(f_alpha_expression_2[3]), globals(), row.to_dict())
    value2_4, value2_5 = eval(str(f_alpha_expression_2[4]), globals(), row.to_dict()), eval(str(f_alpha_expression_2[5]), globals(), row.to_dict())
    value2_6, value2_7 = eval(str(f_alpha_expression_2[6]), globals(), row.to_dict()), eval(str(f_alpha_expression_2[7]), globals(), row.to_dict())
    f_vector2_0.append(value2_0),f_vector2_1.append(value2_1),f_vector2_2.append(value2_2),f_vector2_3.append(value2_3),f_vector2_4.append(value2_4),f_vector2_5.append(value2_5),f_vector2_6.append(value2_6),f_vector2_7.append(value2_7)
f_vector3_0, f_vector3_1, f_vector3_2, f_vector3_3, f_vector3_4, f_vector3_5, f_vector3_6, f_vector3_7 = [], [], [], [], [], [], [], []
for index, row in dr1.iterrows():
    value3_0, value3_1 = eval(str(f_alpha_expression_3[0]), globals(), row.to_dict()), eval(str(f_alpha_expression_3[1]), globals(), row.to_dict())
    value3_2, value3_3 = eval(str(f_alpha_expression_3[2]), globals(), row.to_dict()), eval(str(f_alpha_expression_3[3]), globals(), row.to_dict())
    value3_4, value3_5 = eval(str(f_alpha_expression_3[4]), globals(), row.to_dict()), eval(str(f_alpha_expression_3[5]), globals(), row.to_dict())
    value3_6, value3_7 = eval(str(f_alpha_expression_3[6]), globals(), row.to_dict()), eval(str(f_alpha_expression_3[7]), globals(), row.to_dict())
    f_vector3_0.append(value3_0),f_vector3_1.append(value3_1),f_vector3_2.append(value3_2),f_vector3_3.append(value3_3),f_vector3_4.append(value3_4),f_vector3_5.append(value3_5),f_vector3_6.append(value3_6),f_vector3_7.append(value3_7),
f_vector4_0, f_vector4_1, f_vector4_2, f_vector4_3, f_vector4_4, f_vector4_5, f_vector4_6, f_vector4_7 = [], [], [], [], [], [], [], []
for index, row in dr1.iterrows():
    value4_0, value4_1 = eval(str(f_alpha_expression_4[0]), globals(), row.to_dict()), eval(str(f_alpha_expression_4[1]), globals(), row.to_dict())
    value4_2, value4_3 = eval(str(f_alpha_expression_4[2]), globals(), row.to_dict()), eval(str(f_alpha_expression_4[3]), globals(), row.to_dict())
    value4_4, value4_5 = eval(str(f_alpha_expression_4[4]), globals(), row.to_dict()), eval(str(f_alpha_expression_4[5]), globals(), row.to_dict())
    value4_6, value4_7 = eval(str(f_alpha_expression_4[6]), globals(), row.to_dict()), eval(str(f_alpha_expression_4[7]), globals(), row.to_dict())
    f_vector4_0.append(value4_0),f_vector4_1.append(value4_1),f_vector4_2.append(value4_2),f_vector4_3.append(value4_3),f_vector4_4.append(value4_4),f_vector4_5.append(value4_5),f_vector4_6.append(value4_6),f_vector4_7.append(value4_7)   
f_vector5_0, f_vector5_1, f_vector5_2, f_vector5_3, f_vector5_4, f_vector5_5, f_vector5_6, f_vector5_7 = [], [], [], [], [], [], [], []
for index, row in dr1.iterrows():
    value5_0, value5_1 = eval(str(f_alpha_expression_5[0]), globals(), row.to_dict()), eval(str(f_alpha_expression_5[1]), globals(), row.to_dict())
    value5_2, value5_3 = eval(str(f_alpha_expression_5[2]), globals(), row.to_dict()), eval(str(f_alpha_expression_5[3]), globals(), row.to_dict())
    value5_4, value5_5 = eval(str(f_alpha_expression_5[4]), globals(), row.to_dict()), eval(str(f_alpha_expression_5[5]), globals(), row.to_dict())
    value5_6, value5_7 = eval(str(f_alpha_expression_5[6]), globals(), row.to_dict()), eval(str(f_alpha_expression_5[7]), globals(), row.to_dict())
    f_vector5_0.append(value5_0),f_vector5_1.append(value5_1),f_vector5_2.append(value5_2),f_vector5_3.append(value5_3),f_vector5_4.append(value5_4),f_vector5_5.append(value5_5),f_vector5_6.append(value5_6),f_vector5_7.append(value5_7)    
h1, h2, h3, h4 = list(y_values_dicts_list[0][1].values()), list(y_values_dicts_list[1][1].values()), list(y_values_dicts_list[2][1].values()), list(y_values_dicts_list[3][1].values())
h5, h6, h7, h8 = list(y_values_dicts_list[4][1].values()), list(y_values_dicts_list[5][1].values()), list(y_values_dicts_list[6][1].values()), list(y_values_dicts_list[7][1].values())
y_B501_0, y_B501_1, y_B501_2, y_B501_3, y_B501_4 = np.repeat(h1, rep4), np.repeat(h2, rep4), np.repeat(h3, rep4), np.repeat(h4, rep4), np.repeat(h5, rep4)
y_B501_5, y_B501_6, y_B501_7, y_B502_0, y_B502_1 = np.repeat(h6, rep4), np.repeat(h7, rep4), np.repeat(h8, rep4), np.repeat(h1, rep4), np.repeat(h2, rep4)
y_B502_2, y_B502_3, y_B502_4, y_B502_5, y_B502_6 = np.repeat(h3, rep4), np.repeat(h4, rep4), np.repeat(h5, rep4), np.repeat(h6, rep4), np.repeat(h7, rep4)
y_B502_7, y_B503_0, y_B503_1,y_B503_2, y_B503_3 = np.repeat(h8, rep4), np.repeat(h1, rep4), np.repeat(h2, rep4), np.repeat(h3, rep4), np.repeat(h4, rep4)
y_B503_4,y_B503_5, y_B503_6, y_B503_7, y_B504_0 = np.repeat(h5, rep4), np.repeat(h6, rep4), np.repeat(h7, rep4), np.repeat(h8, rep4), np.repeat(h1, rep4)
y_B504_1, y_B504_2, y_B504_3, y_B504_4, y_B504_5 = np.repeat(h2, rep4), np.repeat(h3, rep4), np.repeat(h4, rep4), np.repeat(h5, rep4), np.repeat(h6, rep4)
y_B504_6, y_B504_7 = np.repeat(h7, rep4), np.repeat(h8, rep4)
y_B505_0, y_B505_1, y_B505_2, y_B505_3, y_B505_4 = np.repeat(h1, rep4), np.repeat(h2, rep4), np.repeat(h3, rep4), np.repeat(h4, rep4), np.repeat(h5, rep4)
y_B505_5, y_B505_6, y_B505_7 = np.repeat(h6, rep4), np.repeat(h7, rep4), np.repeat(h8, rep4)
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
RMSE_5_0, RMSE_5_1 = np.sqrt(mean_squared_error(y_B505_0, f_vector5_0)), np.sqrt(mean_squared_error(y_B505_1, f_vector5_1))
RMSE_5_2, RMSE_5_3 = np.sqrt(mean_squared_error(y_B505_2, f_vector5_2)), np.sqrt(mean_squared_error(y_B505_3, f_vector5_3))
RMSE_5_4, RMSE_5_5 = np.sqrt(mean_squared_error(y_B505_4, f_vector5_4)), np.sqrt(mean_squared_error(y_B505_5, f_vector5_5))
RMSE_5_6, RMSE_5_7 = np.sqrt(mean_squared_error(y_B505_6, f_vector5_6)), np.sqrt(mean_squared_error(y_B505_7, f_vector5_7))
rmse_values = {
    "B501": [RMSE_1_0, RMSE_1_1, RMSE_1_2, RMSE_1_3, RMSE_1_4, RMSE_1_5, RMSE_1_6, RMSE_1_7],
    "B502": [RMSE_2_0, RMSE_2_1, RMSE_2_2, RMSE_2_3, RMSE_2_4, RMSE_2_5, RMSE_2_6, RMSE_2_7],
    "B503": [RMSE_3_0, RMSE_3_1, RMSE_3_2, RMSE_3_3, RMSE_3_4, RMSE_3_5, RMSE_3_6, RMSE_3_7],
    "B504": [RMSE_4_0, RMSE_4_1, RMSE_4_2, RMSE_4_3, RMSE_4_4, RMSE_4_5, RMSE_4_6, RMSE_4_7],
    "B505": [RMSE_5_0, RMSE_5_1, RMSE_5_2, RMSE_5_3, RMSE_5_4, RMSE_5_5, RMSE_5_6, RMSE_5_7]
}
for file, rmse_list in rmse_values.items():
    min_rmse = min(rmse_list)
    min_index = rmse_list.index(min_rmse)
rmse_values = [RMSE_1_0, RMSE_1_1, RMSE_1_2, RMSE_1_3, RMSE_1_4, RMSE_1_5, RMSE_1_6, RMSE_1_7,
               RMSE_2_0, RMSE_2_1, RMSE_2_2, RMSE_2_3, RMSE_2_4, RMSE_2_5, RMSE_2_6, RMSE_2_7,
               RMSE_3_0, RMSE_3_1, RMSE_3_2, RMSE_3_3, RMSE_3_4, RMSE_3_5, RMSE_3_6, RMSE_3_7,
               RMSE_4_0, RMSE_4_1, RMSE_4_2, RMSE_4_3, RMSE_4_4, RMSE_4_5, RMSE_4_6, RMSE_4_7,
               RMSE_5_0, RMSE_5_1, RMSE_5_2, RMSE_5_3, RMSE_5_4, RMSE_5_5, RMSE_5_6, RMSE_5_7]
min_rmse = min(rmse_values)
min_index = rmse_values.index(min_rmse)
file_index = min_index // 8 + 1
sub_index = min_index % 8
h_values = [h1, h2, h3, h4, h5, h6, h7, h8]
lambda_values = [10**(-i) for i in range(lamda)]
h_value = h_values[sub_index]
df2 = pd.read_csv('trainingp_data.csv')
m = df2['trajectory'].nunique()
# df2['trajectory'] = df2['trajectory'].replace({1: h_value[0],2: h_value[1], 3: h_value[2], 4: h_value[3], 5: h_value[4]})
df2['trajectory'] = df2['trajectory'].replace({i: h_value[i-1] for i in range(1, m+1)})
X_train = df2.iloc[:, :-1]
y_train = df2.iloc[:, -1]
X_train.to_csv('X_train.csv', index=False)
y_train = y_train.astype(float)
X_train = X_train.astype(float)
def polynomial_kernel(X, Y, degree=d):
    return (1 + np.dot(X, Y.T)) ** degree
param_grid = {'alpha': [0.0000002, 0.000004, 0.00006, 0.0001, 0.001, 0.010, 0.1, 1, 10]}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
kr_model = KernelRidge(kernel=polynomial_kernel)
grid_search = GridSearchCV(kr_model, param_grid, cv=cv, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# print("Best RMSE:", -grid_search.best_score_)
print("")
print("")
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
        params['degree'] = kwargs.get('degree', 4)
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
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12') 
polynomial_kernel = (1 + x1*sp.Symbol('xi1') + x2*sp.Symbol('xi2') + x3*sp.Symbol('xi3') + x4*sp.Symbol('xi4') + x5*sp.Symbol('xi5') + x6*sp.Symbol('xi6') + x7*sp.Symbol('xi7') + x8*sp.Symbol('xi8') + x9*sp.Symbol('xi9') + x10*sp.Symbol('xi10') + x11*sp.Symbol('xi11') + x12*sp.Symbol('xi12'))**4
f_beta = 0
for i in range(len(X_train)):
    f_beta += eta[i] * polynomial_kernel.subs({'xi1': X_train.iloc[i][0], 'xi2': X_train.iloc[i][1], 'xi3': X_train.iloc[i][2], 'xi4': X_train.iloc[i][3], 'xi5': X_train.iloc[i][4], 'xi6': X_train.iloc[i][5], 'xi7': X_train.iloc[i][6], 'xi8': X_train.iloc[i][7], 'xi9': X_train.iloc[i][8], 'xi10': X_train.iloc[i][9], 'xi11': X_train.iloc[i][10], 'xi12': X_train.iloc[i][11]})
candidate_CL = sp.expand(f_beta)
# print("Candidate Conservation Law:")
# sp.pprint(candidate_CL)
print("")
print("")
expanded_result1 = sp.expand(candidate_CL)
coefficients1 = list(expanded_result1.as_coefficients_dict().values())
terms1 = list(expanded_result1.as_coefficients_dict().keys())
filtered_terms1 = [term for coeff, term in zip(coefficients1, terms1) if abs(coeff) > 0.001]
filtered_expression1 = sum(sp.Mul(coeff, term) for coeff, term in zip(coefficients1, terms1) if term in filtered_terms1)
print("Collected f_alpha(x_q):")
sp.pprint(expanded_result1)
print("")
print("")
print("Collected f_alpha(x_q) with terms having coefficients greater than 0.001:")
sp.pprint(filtered_expression1)
print("")
print("")
with open("ud.txt", "w") as file:
    file.write(str(candidate_CL))
with open("ud.txt", "r") as file:
    ud = sp.sympify(file.read())
df3 = pd.read_csv('holdoutp_data5.csv')
traj_len = df3.groupby('trajectory').size()
rep1 = int(round(traj_len.mean()))
expression = sp.lambdify((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), candidate_CL, "numpy")
df3['lamhold'] = expression(df3['x1'], df3['x2'], df3['x3'], df3['x4'], df3['x5'], df3['x6'], df3['x7'], df3['x8'], df3['x9'], df3['x10'], df3['x11'], df3['x12'])
da = {'y0': h_value[0], 'y1': h_value[1], 'y2': h_value[2], 'y3': h_value[3], 'y4': h_value[4]}
df3['Coluh(lamhold)'] = [da[f'y{i}'] for i in range(m) for _ in range(rep1)]
columns_to_compare = [('lamhold', 'Coluh(lamhold)')]
for col1, col2 in columns_to_compare:
    rmse = np.sqrt(mean_squared_error(df3[col1], df3[col2]))
    print(f'Generalisation Error (RMSE): {rmse}')
    print("")
    print("")
with open("ud.txt", "r") as file:
    ud = sp.sympify(file.read())
f = sp.lambdify((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), ud, "numpy")
dat = pd.read_csv('holdoutp_data5.csv') 
trajectories = dat['trajectory'].unique()
total_sum_squared_normalized_functional_value = 0
total_data_points = 0
num_x_variables = 12
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


# ### Sparsification
# 
# $\Psi_{1}=k=x_{11}, \Psi_{2}=\alpha=x_{12}, \Psi_{3} = \dot{x}_1 + \dot{x}_2 + \dot{x}_3 + \dot{x}_4 + \dot{x}_5 = x_{6} + x_{7} + x_{8} + x_{9} + x_{10}$
# 
# We now have to work in the linear space
#   $\mathbb{R}[\Psi_{1},\Psi_{2},\Psi_{3},\Psi_{4}]_{\leq 4}$.  A generating set of it is
#   given by $\Psi_{4}$ and all terms $\Psi_{1}^{d_1}\Psi_{2}^{d_2}\Psi_{3}^{d_3}$ with
#   exponents $d_1,d_2,d_3\in\mathbb{N}_{0}$ satisfying $d_1+d_2+d_3\leq 4$

# In[1082]:


x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12')
filtered_expression8 = filtered_expression1
f1=filtered_expression1
f2 = filtered_expression1
expr1 = filtered_expression8
x12 = sp.symbols('x12')
exprpk =  x12
polynomialpk = sp.Poly(exprpk, x12)
terms_pk = polynomialpk.monoms()
coeffs_pk = polynomialpk.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk = {sp.Mul(*[s**e for s, e in zip([x12], term)]): coeff
                          for term, coeff in zip(terms_pk, coeffs_pk)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpk2 =  x12**2
polynomialpk2 = sp.Poly(exprpk2, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pk2 = polynomialpk2.monoms()
coeffs_pk2 = polynomialpk2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk2 = {sp.Mul(*[s**e for s, e in zip([x12], term)]): coeff
                          for term, coeff in zip(terms_pk2, coeffs_pk2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpk3 =  x12**3
polynomialpk3 = sp.Poly(exprpk3, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pk3 = polynomialpk3.monoms()
coeffs_pk3 = polynomialpk3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk3 = {sp.Mul(*[s**e for s, e in zip([x12], term)]): coeff
                          for term, coeff in zip(terms_pk3, coeffs_pk3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpk4 =  x12**4
polynomialpk4 = sp.Poly(exprpk4, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pk4 = polynomialpk4.monoms()
coeffs_pk4 = polynomialpk4.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk4 = {sp.Mul(*[s**e for s, e in zip([x12], term)]): coeff
                          for term, coeff in zip(terms_pk4, coeffs_pk4)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk4.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpka =  x11*x12
polynomialpka = sp.Poly(exprpka, x11, x12)
terms_pka = polynomialpka.monoms()
coeffs_pka = polynomialpka.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pka = {sp.Mul(*[s**e for s, e in zip([x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pka, coeffs_pka)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pka.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpka2 =  x11*x12**2
polynomialpka2 = sp.Poly(exprpka2, x11, x12)
terms_pka2 = polynomialpka2.monoms()
coeffs_pka2 = polynomialpka2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pka2 = {sp.Mul(*[s**e for s, e in zip([x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pka2, coeffs_pka2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pka2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpka3 =  x11*x12**3
polynomialpka3 = sp.Poly(exprpka3, x11, x12)
terms_pka3 = polynomialpka3.monoms()
coeffs_pka3 = polynomialpka3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pka3 = {sp.Mul(*[s**e for s, e in zip([x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pka3, coeffs_pka3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pka3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpk2a2 =  x11**2*x12**2
polynomialpk2a2 = sp.Poly(exprpk2a2, x11, x12)
terms_pk2a2 = polynomialpk2a2.monoms()
coeffs_pk2a2 = polynomialpk2a2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk2a2 = {sp.Mul(*[s**e for s, e in zip([x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pk2a2, coeffs_pk2a2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk2a2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpk2a =  x11**2*x12
polynomialpk2a = sp.Poly(exprpk2a, x11, x12)
terms_pk2a = polynomialpk2a.monoms()
coeffs_pk2a = polynomialpk2a.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk2a = {sp.Mul(*[s**e for s, e in zip([x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pk2a, coeffs_pk2a)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk2a.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpk2a3 =  x11**3*x12
polynomialpk2a3 = sp.Poly(exprpk2a3, x11, x12)
terms_pk2a3 = polynomialpk2a3.monoms()
coeffs_pk2a3 = polynomialpk2a3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk2a3 = {sp.Mul(*[s**e for s, e in zip([x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pk2a3, coeffs_pk2a3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk2a3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpa =  x11
polynomialpa = sp.Poly(exprpa, x11)
terms_pa = polynomialpa.monoms()
coeffs_pa = polynomialpa.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pa = {sp.Mul(*[s**e for s, e in zip([x11], term)]): coeff
                          for term, coeff in zip(terms_pa, coeffs_pa)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pa.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpa2 =  x11**2
polynomialpa2 = sp.Poly(exprpa2, x11)
terms_pa2 = polynomialpa2.monoms()
coeffs_pa2 = polynomialpa2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pa2 = {sp.Mul(*[s**e for s, e in zip([x11], term)]): coeff
                          for term, coeff in zip(terms_pa2, coeffs_pa2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pa2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpa3 =  x11**3
polynomialpa3 = sp.Poly(exprpa3, x11)
terms_pa3 = polynomialpa3.monoms()
coeffs_pa3 = polynomialpa3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pa3 = {sp.Mul(*[s**e for s, e in zip([x11], term)]): coeff
                          for term, coeff in zip(terms_pa3, coeffs_pa3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pa3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpa4 =  x11**4
polynomialpa4 = sp.Poly(exprpa4, x11)
terms_pa4 = polynomialpa4.monoms()
coeffs_pa4 = polynomialpa4.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pa4 = {sp.Mul(*[s**e for s, e in zip([x11], term)]): coeff
                          for term, coeff in zip(terms_pa4, coeffs_pa4)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pa4.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprps =  x6 + x7 + x8 + x9 + x10
polynomialps = sp.Poly(exprps, x6, x7, x8, x9, x10)
terms_ps = polynomialps.monoms()
coeffs_ps = polynomialps.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps = {sp.Mul(*[s**e for s, e in zip([x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps, coeffs_ps)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprps2 =  (x6 + x7 + x8 + x9 + x10)**2
polynomialps2 = sp.Poly(exprps2, x6, x7, x8, x9, x10)
terms_ps2 = polynomialps2.monoms()
coeffs_ps2 = polynomialps2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps2 = {sp.Mul(*[s**e for s, e in zip([x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps2, coeffs_ps2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprps3 =  (x6 + x7 + x8 + x9 + x10)**3
polynomialps3 = sp.Poly(exprps3, x6, x7, x8, x9, x10)
terms_ps3 = polynomialps3.monoms()
coeffs_ps3 = polynomialps3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps3 = {sp.Mul(*[s**e for s, e in zip([x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps3, coeffs_ps3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprps4 =  (x6 + x7 + x8 + x9 + x10)**4
polynomialps4 = sp.Poly(exprps4, x6, x7, x8, x9, x10)
terms_ps4 = polynomialps4.monoms()
coeffs_ps4 = polynomialps4.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps4 = {sp.Mul(*[s**e for s, e in zip([x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps4, coeffs_ps4)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps4.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpsk =  x12*(x6 + x7 + x8 + x9 + x10)
polynomialpsk = sp.Poly(exprpsk, x12, x6, x7, x8, x9, x10)
terms_psk = polynomialpsk.monoms()
coeffs_psk = polynomialpsk.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psk = {sp.Mul(*[s**e for s, e in zip([x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psk, coeffs_psk)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psk.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpsk2 =  x12**2*(x6 + x7 + x8 + x9 + x10)
polynomialpsk2 = sp.Poly(exprpsk2, x12, x6, x7, x8, x9, x10)
terms_psk2 = polynomialpsk2.monoms()
coeffs_psk2 = polynomialpsk2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psk2 = {sp.Mul(*[s**e for s, e in zip([x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psk2, coeffs_psk2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psk2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpsk3 =  x12**3*(x6 + x7 + x8 + x9 + x10)
polynomialpsk3 = sp.Poly(exprpsk3, x12, x6, x7, x8, x9, x10)
terms_psk3 = polynomialpsk3.monoms()
coeffs_psk3 = polynomialpsk3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psk3 = {sp.Mul(*[s**e for s, e in zip([x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psk3, coeffs_psk3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psk3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpsa =  x11*(x6 + x7 + x8 + x9 + x10)
polynomialpsa = sp.Poly(exprpsa, x11, x6, x7, x8, x9, x10)
terms_psa = polynomialpsa.monoms()
coeffs_psa = polynomialpsa.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psa = {sp.Mul(*[s**e for s, e in zip([x11, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psa, coeffs_psa)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psa.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpsa2 =  x11**2*(x6 + x7 + x8 + x9 + x10)
polynomialpsa2 = sp.Poly(exprpsa2, x11, x6, x7, x8, x9, x10)
terms_psa2 = polynomialpsa2.monoms()
coeffs_psa2 = polynomialpsa2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psa2 = {sp.Mul(*[s**e for s, e in zip([x11, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psa2, coeffs_psa2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psa2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpsa3 =  x11**3*(x6 + x7 + x8 + x9 + x10)
polynomialpsa3 = sp.Poly(exprpsa3, x11, x6, x7, x8, x9, x10)
terms_psa3 = polynomialpsa3.monoms()
coeffs_psa3 = polynomialpsa3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psa3 = {sp.Mul(*[s**e for s, e in zip([x11, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psa3, coeffs_psa3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psa3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprps2k =  x12*(x6 + x7 + x8 + x9 + x10)**2
polynomialps2k = sp.Poly(exprps2k, x12, x6, x7, x8, x9, x10)
terms_ps2k = polynomialps2k.monoms()
coeffs_ps2k = polynomialps2k.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps2k = {sp.Mul(*[s**e for s, e in zip([x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps2k, coeffs_ps2k)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps2k.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprps2k2 =  x12**2*(x6 + x7 + x8 + x9 + x10)**2
polynomialps2k2 = sp.Poly(exprps2k2, x12, x6, x7, x8, x9, x10)
terms_ps2k2 = polynomialps2k2.monoms()
coeffs_ps2k2 = polynomialps2k2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps2k2 = {sp.Mul(*[s**e for s, e in zip([x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps2k2, coeffs_ps2k2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps2k2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprps3k =  x12*(x6 + x7 + x8 + x9 + x10)**3
polynomialps3k = sp.Poly(exprps3k, x12, x6, x7, x8, x9, x10)
terms_ps3k = polynomialps3k.monoms()
coeffs_ps3k = polynomialps3k.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps3k = {sp.Mul(*[s**e for s, e in zip([x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps3k, coeffs_ps3k)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps3k.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprps2a =  x11*(x6 + x7 + x8 + x9 + x10)**2
polynomialps2a = sp.Poly(exprps2a, x11, x6, x7, x8, x9, x10)
terms_ps2a = polynomialps2a.monoms()
coeffs_ps2a = polynomialps2a.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps2a = {sp.Mul(*[s**e for s, e in zip([x11, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps2a, coeffs_ps2a)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps2a.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprps2a2 =  x11**2*(x6 + x7 + x8 + x9 + x10)**2
polynomialps2a2 = sp.Poly(exprps2a2, x11, x6, x7, x8, x9, x10)
terms_ps2a2 = polynomialps2a2.monoms()
coeffs_ps2a2 = polynomialps2a2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps2a2 = {sp.Mul(*[s**e for s, e in zip([x11, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps2a2, coeffs_ps2a2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps2a2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprps3a =  x11*(x6 + x7 + x8 + x9 + x10)**3
polynomialps3a = sp.Poly(exprps3a, x11, x6, x7, x8, x9, x10)
terms_ps3a = polynomialps3a.monoms()
coeffs_ps3a = polynomialps3a.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps3a = {sp.Mul(*[s**e for s, e in zip([x11, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps3a, coeffs_ps3a)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps3a.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpsak =  x11*x12*(x6 + x7 + x8 + x9 + x10)
polynomialpsak = sp.Poly(exprpsak, x11, x12, x6, x7, x8, x9, x10)
terms_psak = polynomialpsak.monoms()
coeffs_psak = polynomialpsak.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psak = {sp.Mul(*[s**e for s, e in zip([x11, x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psak, coeffs_psak)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psak.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprps2ak =  x11*x12*(x6 + x7 + x8 + x9 + x10)**2
polynomialps2ak = sp.Poly(exprps2ak, x11, x12, x6, x7, x8, x9, x10)
terms_ps2ak = polynomialps2ak.monoms()
coeffs_ps2ak = polynomialps2ak.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps2ak = {sp.Mul(*[s**e for s, e in zip([x11, x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps2ak, coeffs_ps2ak)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps2ak.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpsak2 =  x11**2*x12*(x6 + x7 + x8 + x9 + x10)
polynomialpsak2 = sp.Poly(exprpsak2, x11, x12, x6, x7, x8, x9, x10)
terms_psak2 = polynomialpsak2.monoms()
coeffs_psak2 = polynomialpsak2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psak2 = {sp.Mul(*[s**e for s, e in zip([x11, x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psak2, coeffs_psak2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psak2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
exprpsa2k =  x11*x12**2*(x6 + x7 + x8 + x9 + x10)
polynomialpsa2k = sp.Poly(exprpsa2k, x11, x12, x6, x7, x8, x9, x10)
terms_psa2k = polynomialpsa2k.monoms()
coeffs_psa2k = polynomialpsa2k.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psa2k = {sp.Mul(*[s**e for s, e in zip([x11, x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psa2k, coeffs_psa2k)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psa2k.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")


# In[1083]:



polynomialpk = sp.Poly(exprpk, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pk = polynomialpk.monoms()
coeffs_pk = polynomialpk.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk = {sp.Mul(*[s**e for s, e in zip([x12], term)]): coeff
                          for term, coeff in zip(terms_pk, coeffs_pk)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk

polynomialpk2 = sp.Poly(exprpk2, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pk2 = polynomialpk2.monoms()
coeffs_pk2 = polynomialpk2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk2 = {sp.Mul(*[s**e for s, e in zip([x12], term)]): coeff
                          for term, coeff in zip(terms_pk2, coeffs_pk2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk2 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk2

polynomialpk3 = sp.Poly(exprpk3, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pk3 = polynomialpk3.monoms()
coeffs_pk3 = polynomialpk3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk3 = {sp.Mul(*[s**e for s, e in zip([x12], term)]): coeff
                          for term, coeff in zip(terms_pk3, coeffs_pk3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk3 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk3

polynomialpk4 = sp.Poly(exprpk4, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pk4 = polynomialpk4.monoms()
coeffs_pk4 = polynomialpk4.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk4 = {sp.Mul(*[s**e for s, e in zip([x12], term)]): coeff
                          for term, coeff in zip(terms_pk4, coeffs_pk4)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk4.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk4 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk4

polynomialpka = sp.Poly(exprpka, x11, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pka = polynomialpka.monoms()
coeffs_pka = polynomialpka.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pka = {sp.Mul(*[s**e for s, e in zip([x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pka, coeffs_pka)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pka.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk5 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk5
polynomialpka2 = sp.Poly(exprpka2, x11, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pka2 = polynomialpka2.monoms()
coeffs_pka2 = polynomialpka2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pka2 = {sp.Mul(*[s**e for s, e in zip([x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pka2, coeffs_pka2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pka2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk6 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk6
polynomialpka3 = sp.Poly(exprpka3, x11, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pka3 = polynomialpka3.monoms()
coeffs_pka3 = polynomialpka3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pka3 = {sp.Mul(*[s**e for s, e in zip([x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pka3, coeffs_pka3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pka3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk7 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk7
polynomialpk2a2 = sp.Poly(exprpk2a2, x11, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pk2a2 = polynomialpk2a2.monoms()
coeffs_pk2a2 = polynomialpk2a2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk2a2 = {sp.Mul(*[s**e for s, e in zip([x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pk2a2, coeffs_pk2a2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk2a2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk8 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk8
polynomialpk2a = sp.Poly(exprpk2a, x11, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pk2a = polynomialpk2a.monoms()
coeffs_pk2a = polynomialpk2a.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk2a = {sp.Mul(*[s**e for s, e in zip([x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pk2a, coeffs_pk2a)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk2a.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk9 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk9
polynomialpk2a3 = sp.Poly(exprpk2a3, x11, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pk2a3 = polynomialpk2a3.monoms()
coeffs_pk2a3 = polynomialpk2a3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pk2a3 = {sp.Mul(*[s**e for s, e in zip([x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pk2a3, coeffs_pk2a3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pk2a3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk10 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk10
polynomialpa = sp.Poly(exprpa, x11)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pa = polynomialpa.monoms()
coeffs_pa = polynomialpa.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pa = {sp.Mul(*[s**e for s, e in zip([x11], term)]): coeff
                          for term, coeff in zip(terms_pa, coeffs_pa)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pa.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk11 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk11
polynomialpa2 = sp.Poly(exprpa2, x11)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pa2 = polynomialpa2.monoms()
coeffs_pa2 = polynomialpa2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pa2 = {sp.Mul(*[s**e for s, e in zip([x11], term)]): coeff
                          for term, coeff in zip(terms_pa2, coeffs_pa2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pa2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk12 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk12
polynomialpa3 = sp.Poly(exprpa3, x11)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pa3 = polynomialpa3.monoms()
coeffs_pa3 = polynomialpa3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pa3 = {sp.Mul(*[s**e for s, e in zip([x11], term)]): coeff
                          for term, coeff in zip(terms_pa3, coeffs_pa3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pa3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk13 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk13
polynomialpa4 = sp.Poly(exprpa4, x11)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pa4 = polynomialpa4.monoms()
coeffs_pa4 = polynomialpa4.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pa4 = {sp.Mul(*[s**e for s, e in zip([x11], term)]): coeff
                          for term, coeff in zip(terms_pa4, coeffs_pa4)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pa4.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk14 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk14
polynomialps = sp.Poly(exprps, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_ps = polynomialps.monoms()
coeffs_ps = polynomialps.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps = {sp.Mul(*[s**e for s, e in zip([x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps, coeffs_ps)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk15 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk15
polynomialps2 = sp.Poly(exprps2, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_ps2 = polynomialps2.monoms()
coeffs_ps2 = polynomialps2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps2 = {sp.Mul(*[s**e for s, e in zip([x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps2, coeffs_ps2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk16 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk16
polynomialps3 = sp.Poly(exprps3, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_ps3 = polynomialps3.monoms()
coeffs_ps3 = polynomialps3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps3 = {sp.Mul(*[s**e for s, e in zip([x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps3, coeffs_ps3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk17 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk17
polynomialps4 = sp.Poly(exprps4, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_ps4 = polynomialps4.monoms()
coeffs_ps4 = polynomialps4.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps4 = {sp.Mul(*[s**e for s, e in zip([x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps4, coeffs_ps4)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps4.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk18 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk18

polynomialpsk = sp.Poly(exprpsk, x12, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_psk = polynomialpsk.monoms()
coeffs_psk = polynomialpsk.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psk = {sp.Mul(*[s**e for s, e in zip([x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psk, coeffs_psk)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psk.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk19 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk19
polynomialpsk2 = sp.Poly(exprpsk2, x12, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_psk2 = polynomialpsk2.monoms()
coeffs_psk2 = polynomialpsk2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psk2 = {sp.Mul(*[s**e for s, e in zip([x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psk2, coeffs_psk2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psk2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk20 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk20
polynomialpsk3 = sp.Poly(exprpsk3, x12, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_psk3 = polynomialpsk3.monoms()
coeffs_psk3 = polynomialpsk3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psk3 = {sp.Mul(*[s**e for s, e in zip([x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psk3, coeffs_psk3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psk3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk21 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk21
polynomialpsa = sp.Poly(exprpsa, x11, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_psa = polynomialpsa.monoms()
coeffs_psa = polynomialpsa.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psa = {sp.Mul(*[s**e for s, e in zip([x11, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psa, coeffs_psa)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psa.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk22 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk22
polynomialpsa2 = sp.Poly(exprpsa2, x11, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_psa2 = polynomialpsa2.monoms()
coeffs_psa2 = polynomialpsa2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psa2 = {sp.Mul(*[s**e for s, e in zip([x11, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psa2, coeffs_psa2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psa2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk23 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk23
polynomialpsa3 = sp.Poly(exprpsa3, x11, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_psa3 = polynomialpsa3.monoms()
coeffs_psa3 = polynomialpsa3.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psa3 = {sp.Mul(*[s**e for s, e in zip([x11, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psa3, coeffs_psa3)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psa3.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk24 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk24
polynomialps2k = sp.Poly(exprps2k, x12, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_ps2k = polynomialps2k.monoms()
coeffs_ps2k = polynomialps2k.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps2k = {sp.Mul(*[s**e for s, e in zip([x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps2k, coeffs_ps2k)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps2k.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk25 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk25
polynomialps2k2 = sp.Poly(exprps2k2, x12, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_ps2k2 = polynomialps2k2.monoms()
coeffs_ps2k2 = polynomialps2k2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps2k2 = {sp.Mul(*[s**e for s, e in zip([x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps2k2, coeffs_ps2k2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps2k2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk26 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk26
polynomialps3k = sp.Poly(exprps3k, x12, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_ps3k = polynomialps3k.monoms()
coeffs_ps3k = polynomialps3k.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps3k = {sp.Mul(*[s**e for s, e in zip([x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps3k, coeffs_ps3k)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps3k.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk27 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk27
polynomialps2a = sp.Poly(exprps2a, x11, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_ps2a = polynomialps2a.monoms()
coeffs_ps2a = polynomialps2a.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps2a = {sp.Mul(*[s**e for s, e in zip([x11, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps2a, coeffs_ps2a)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps2a.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk28 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk28
polynomialps2a2 = sp.Poly(exprps2a2, x11, x6, x7, x8, x9, x10)
terms_ps2a2 = polynomialps2a2.monoms()
coeffs_ps2a2 = polynomialps2a2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps2a2 = {sp.Mul(*[s**e for s, e in zip([x11, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps2a2, coeffs_ps2a2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps2a2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk29 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk29
polynomialps3a = sp.Poly(exprps3a, x11, x6, x7, x8, x9, x10)
terms_ps3a = polynomialps3a.monoms()
coeffs_ps3a = polynomialps3a.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps3a = {sp.Mul(*[s**e for s, e in zip([x11, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps3a, coeffs_ps3a)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps3a.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk30 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk30
polynomialpsak = sp.Poly(exprpsak, x11, x12, x6, x7, x8, x9, x10)
terms_psak = polynomialpsak.monoms()
coeffs_psak = polynomialpsak.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psak = {sp.Mul(*[s**e for s, e in zip([x11, x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psak, coeffs_psak)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psak.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk31 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk31
polynomialps2ak = sp.Poly(exprps2ak, x11, x12, x6, x7, x8, x9, x10)
terms_ps2ak = polynomialps2ak.monoms()
coeffs_ps2ak = polynomialps2ak.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps2ak = {sp.Mul(*[s**e for s, e in zip([x11, x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps2ak, coeffs_ps2ak)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps2ak.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk32 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk32
polynomialpsak2 = sp.Poly(exprpsak2, x11, x12, x6, x7, x8, x9, x10)
terms_psak2 = polynomialpsak2.monoms()
coeffs_psak2 = polynomialpsak2.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psak2 = {sp.Mul(*[s**e for s, e in zip([x11, x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psak2, coeffs_psak2)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psak2.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk33 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk33
polynomialpsa2k = sp.Poly(exprpsa2k, x11, x12, x6, x7, x8, x9, x10)
terms_psa2k = polynomialpsa2k.monoms()
coeffs_psa2k = polynomialpsa2k.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_psa2k = {sp.Mul(*[s**e for s, e in zip([x11, x12, x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_psa2k, coeffs_psa2k)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_psa2k.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk34 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixk34
filtered_expression8 = (filtered_expression1)
exprpkf = filtered_expression8
polynomialpkf = sp.Poly(exprpkf, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pkf = polynomialpkf.monoms()
coeffs_pkf = polynomialpkf.coeffs()
terms_with_coeffs_pkf = {
    sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)]): coeff
    for term, coeff in zip(terms_pkf, coeffs_pkf)
}
# Sort terms by total degree in ascending order
sorted_terms_with_coeffs = sorted(
    terms_with_coeffs_pkf.items(),
    key=lambda item: sum(sp.degree_list(item[0]))
)
terms_with_coeffs1 = dict(sorted_terms_with_coeffs)
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
vp1 = list(terms_with_coeffs1.keys())
print("vp1 =", vp1)


# In[1084]:



x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12')
filtered_expression8 = filtered_expression1
expr1 = filtered_expression8
exprpkf = filtered_expression8
polynomialpkf = sp.Poly(exprpkf, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pkf = polynomialpkf.monoms()
coeffs_pkf = polynomialpkf.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pkf = {sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pkf, coeffs_pkf)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pkf.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
for term, coeff in terms_with_coeffs1.items():
    print(f"{term}: {coeff}")
vp1 = list(terms_with_coeffs1.keys())
print("vp1 =", vp1)

polynomialpkf = sp.Poly(exprpkf, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pkf = polynomialpkf.monoms()
coeffs_pkf = polynomialpkf.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pkf = {sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pkf, coeffs_pkf)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pkf.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixkf = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixkf


# In[1085]:


combined_matrixp6 = sp.Matrix.hstack(coeffs_column_matrixk, coeffs_column_matrixk2, coeffs_column_matrixk3, coeffs_column_matrixk4, coeffs_column_matrixk5, coeffs_column_matrixk6, coeffs_column_matrixk7, coeffs_column_matrixk8, coeffs_column_matrixk9, coeffs_column_matrixk10, coeffs_column_matrixk11, coeffs_column_matrixk12, coeffs_column_matrixk13, coeffs_column_matrixk14, coeffs_column_matrixk15, coeffs_column_matrixk16, coeffs_column_matrixk17, coeffs_column_matrixk18, coeffs_column_matrixk19, coeffs_column_matrixk20, coeffs_column_matrixk21, coeffs_column_matrixk22, coeffs_column_matrixk23, coeffs_column_matrixk24, coeffs_column_matrixk25, coeffs_column_matrixk26, coeffs_column_matrixk27, coeffs_column_matrixk28, coeffs_column_matrixk29, coeffs_column_matrixk30, coeffs_column_matrixk31, coeffs_column_matrixk32, coeffs_column_matrixk33, coeffs_column_matrixk34, coeffs_column_matrixkf)
sp.pprint(combined_matrixp6)
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12')
filtered_expression8 = filtered_expression1
expr1 = filtered_expression8
exprpkf = filtered_expression8
polynomialpkf = sp.Poly(exprpkf, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pkf = polynomialpkf.monoms()
coeffs_pkf = polynomialpkf.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pkf = {sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pkf, coeffs_pkf)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pkf.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
vp1 = list(terms_with_coeffs1.keys())
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12')
filtered_expression8 = filtered_expression1
expr1 = filtered_expression8
exprpkf = filtered_expression8
polynomialpkf = sp.Poly(exprpkf, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pkf = polynomialpkf.monoms()
coeffs_pkf = polynomialpkf.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pkf = {sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pkf, coeffs_pkf)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pkf.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
vp1 = list(terms_with_coeffs1.keys())
polynomialpkf = sp.Poly(exprpkf, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pkf = polynomialpkf.monoms()
coeffs_pkf = polynomialpkf.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pkf = {sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pkf, coeffs_pkf)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pkf.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixkf = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
coeffs_column_matrixkf
exprps =  x6 + x7 + x8 + x9 + x10
polynomialps = sp.Poly(exprps, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_ps = polynomialps.monoms()
coeffs_ps = polynomialps.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps = {sp.Mul(*[s**e for s, e in zip([x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps, coeffs_ps)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
polynomialpkf = sp.Poly(exprpkf, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_pkf = polynomialpkf.monoms()
coeffs_pkf = polynomialpkf.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_pkf = {sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)]): coeff
                          for term, coeff in zip(terms_pkf, coeffs_pkf)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_pkf.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixkf = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
polynomialps = sp.Poly(exprps, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_ps = polynomialps.monoms()
coeffs_ps = polynomialps.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps = {sp.Mul(*[s**e for s, e in zip([x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps, coeffs_ps)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
polynomialps = sp.Poly(exprps, x6, x7, x8, x9, x10)
polynomial1 = sp.Poly(expr1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
terms_ps = polynomialps.monoms()
coeffs_ps = polynomialps.coeffs()
terms1 = polynomial1.monoms()
coeffs1 = polynomial1.coeffs()
terms_with_coeffs_ps = {sp.Mul(*[s**e for s, e in zip([x6, x7, x8, x9, x10], term)]): coeff
                          for term, coeff in zip(terms_ps, coeffs_ps)}
terms_with_coeffs1 = {}
for term in terms1:
    term_expr = sp.Mul(*[s**e for s, e in zip([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], term)])
    coeff = terms_with_coeffs_ps.get(term_expr, 0)
    terms_with_coeffs1[term_expr] = coeff
coeffs_column_matrixk15 = sp.Matrix([coeff for coeff in terms_with_coeffs1.values()])
combined_matrixp6 = sp.Matrix.hstack(coeffs_column_matrixk15, coeffs_column_matrixkf)
C = combined_matrixp6
a = cp.Variable(2)
Ca = C @ a
objective = cp.Minimize(cp.norm1(Ca))
constraints = [
    cp.sum(a) == 1
]
problem = cp.Problem(objective, constraints)
problem.solve()
a_optimal = a.value
C = combined_matrixp6
a = cp.Variable(2)
Ca = C @ a
objective = cp.Minimize(cp.norm1(Ca))
constraints = [cp.sum(a) == 1]
problem = cp.Problem(objective, constraints)
problem.solve()
a_optimal = a.value
resultp = np.dot(C, a_optimal)
formatted_resultp = ', '.join(map(str, resultp))
ent1 = np.concatenate([np.ones(1), np.zeros(1)])
e1 = ent1.reshape(2, 1)
C1 = combined_matrixp6 
a1 = cp.Variable(2)
C1a1 = C @ a
objective1 = cp.Minimize(cp.norm1(C1a1))
epsilon1 = 1e-6 
constraints1 = [
    cp.sum(a1) == 1,
    e1.T @ a >= epsilon1
]
problem1 = cp.Problem(objective1, constraints1)
problem1.solve()
a1_optimal = a1.value
resultp1 = np.dot(C1, a1_optimal)
formatted_resultp1 = ', '.join(map(str, resultp1))
C = combined_matrixp6
a = a_optimal
resultp = np.dot(C, a)
formatted_resultp = ', '.join(map(str, resultp))
v3_str = '[' + ', '.join(f'{x:.15g}' for x in resultp) + ']'
v3_list = ast.literal_eval(v3_str)
v3_sympy = [sp.Float(x) for x in v3_list]
v1 = vp1
dot_product1 = sum(sp.Mul(v1_i, v3_i) for v1_i, v3_i in zip(v1, v3_sympy))
dot_product1_simplified = sp.simplify(dot_product1)
coefficients = list(dot_product1_simplified.as_coefficients_dict().values())
terms = list(dot_product1_simplified.as_coefficients_dict().keys())
filtered_terms = [term for coeff, term in zip(coefficients, terms) if abs(coeff) > 0.00001]
filtered_expression = sum(sp.Mul(coeff, term) for coeff, term in zip(coefficients, terms) if term in filtered_terms)
print("")
print("")
print("")
print("Expressions greater than 0.00001:")
sp.pprint(filtered_expression)


# In[1087]:


num = len(f1.as_ordered_terms())
print("Number of terms:")
print(num)
num = len(f2.as_ordered_terms())
print(num)
num = len(filtered_expression.as_ordered_terms())
print(num)


# In[ ]:




