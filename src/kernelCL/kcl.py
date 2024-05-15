import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import csv
import itertools
import os
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    KFold,
)
from sklearn.kernel_ridge import KernelRidge
from sympy import symbols, simplify, lambdify, Function, diff
from sklearn.metrics import mean_squared_error
import math
from scipy.stats import pearsonr
import sys

if __name__ == "__main__":
    datafiles = sys.argv[1:]
    if len(datafiles) == 0:
        print("No trajectory data provided.")
        sys.exit(1)

    m = len(datafiles)
    nsplits = 5
    training_frac = 0.8

    # load trajectories
    trajs = []
    for datafile in datafiles:
        traj = np.loadtxt(datafile, delimiter=",", skiprows=1)
        trajs.append(traj)

    # split data into training and holdout sets
    datasets = [list() for _ in range(nsplits)]
    holdout = []
    training = []
    for traj in trajs:
        npts = int(traj.shape[0] * training_frac) // nsplits
        np.random.shuffle(traj)

        for split in range(nsplits):
            part = traj[split * npts : (split + 1) * npts, :]
            datasets[split].append(part)

        ntrain = int(traj.shape[0] * training_frac)
        training.append(traj[:ntrain, :])
        holdout.append(traj[ntrain:, :])
    datasets = [np.vstack(ds) for ds in datasets]  # B
    holdout = np.vstack(holdout)
    training = np.vstack(training)  # X_train

    def compute_kernel_matrix(X, c, d):
        n = X.shape[0]
        K = (c + np.dot(X, X.T)) ** d
        return K

    def solve_for_lambda(data, c, d, lambda_value, m):
        rep = len(data) // m
        K = compute_kernel_matrix(data, c, d)
        I = np.eye(K.shape[0])
        K_with_I = K + lambda_value * I
        K_with_I_inv = np.linalg.inv(K_with_I)
        num_x_variables = data.shape[
            1
        ]  # Indicates the number of variable, which means we have 4 variables from x1 to x4
        num_y_variables = m  # number of trajectory
        x_symbols = [sp.symbols(f"x{i}") for i in range(1, num_x_variables + 1)]
        y_symbols = [sp.symbols(f"y{i}") for i in range(num_y_variables)]
        y0 = sp.symbols("y0")
        y_pattern = [sp.symbols(f"y{i}") for i in range(m)]
        y_repeated = np.repeat(y_pattern, rep, axis=0)
        y = sp.Matrix([sp.symbols(f"y{i}") for i in range(m)])
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
        variables = [sp.symbols(f"y{i}") for i in range(1, m)]
        for i in range(len(y_pattern) - 1):
            equation = sp.Eq(B[i + 1], 0)
            system_of_equations.append(equation)
        solution = sp.solve(system_of_equations, variables)
        solution_with_y0 = {**{"y0": 1}, **solution}
        return {"W": W1, "solution": solution_with_y0}

    def add_matrices(mat1, mat2, mat3, mat4):
        return mat1 + mat2 + mat3 + mat4

    lvals = [10 ** (-i) for i in range(8)]
    c, d = 1, 1  # Adjust the values of variable "d" according to the required degree.

    solutions = [
        [solve_for_lambda(dataset, c, d, l, m) for l in lvals]
        for dataset in datasets[:-1]
    ]

    solutions_combined = []
    for i, (w1, w2, w3, w4) in enumerate(zip(*solutions)):
        added_matrix = add_matrices(w1["W"], w2["W"], w3["W"], w4["W"])
        all_symbols = w1["W"].free_symbols
        set_solution = {"y0": 1}
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
                values = [
                    row[var] ** k if var != "1" else 1 for var, k in zip(variables, ks)
                ]
                term_value = coefficient * math.prod(values)
                expansions.append(term_value)
        return expansions[::-1]

    def multilinear_expansion1(variables1, n1):
        expansions1 = []
        for ks in itertools.product(range(n1 + 1), repeat=len(variables1)):
            if sum(ks) == n1:
                terms1 = [
                    f"{var}**{k}" if k != 0 else f"{var}"
                    for var, k in zip(variables1, ks)
                    if k != 0
                ]
                term1 = " * ".join(terms1)
                expansions1.append(term1)
        return expansions1[::-1]

    def generate_inner_products(coefficients, terms):
        inner_products = [f"{c}*{t}" for c, t in zip(coefficients, terms)]
        return inner_products

    def solve_for_lambda(data, c, d, lambda_value, y_values):
        K = compute_kernel_matrix(data, c, d)
        I = np.eye(K.shape[0])
        K_with_I = K + lambda_value * I
        K_with_I_inv = np.linalg.inv(K_with_I)
        y_repeated = np.repeat(y_values, len(data) // len(y_values))
        alpha_sym = K_with_I_inv @ y_repeated
        return K_with_I_inv, alpha_sym

    def process_dataset(dataset, lambda_values, y_values_dicts):
        data = pd.DataFrame(
            dataset, columns=[f"x{_}" for _ in range(1, dataset.shape[1] + 1)]
        )
        variables = ["c", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
        n = 1  # degree of the polynomial kernel
        data["c"] = 1
        variables1 = ["1", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
        n1 = 1  # degree of the polynomial kernel
        all_entries = []
        f_alpha_expression_list = []
        data_np = dataset
        for index, row in data.iterrows():
            result = multilinear_expansion(variables, n, row)
            result1 = multilinear_expansion1(variables1, n1)
            expressions = generate_inner_products(result, result1)
            all_entries.append(expressions)
        for lambda_val, y_values_dict in zip(lambda_values, y_values_dicts):
            c = 1.0
            d = 1  # degree of the polynomial kernel
            K_with_I_inv, alpha_sym = solve_for_lambda(
                data_np, c, d, lambda_val, list(y_values_dict.values())
            )
            total_sum = 0
            #         print(f"File: {file_path}, Lambda = 10^(-{lambda_values.index(lambda_val)}), y_values_dict: {y_values_dict}:")
            for entry_index, (entry, alpha) in enumerate(
                zip(all_entries, alpha_sym), 1
            ):
                entry_sum = 0
                for term in entry:
                    result = alpha * sp.sympify(term)
                    entry_sum += result
                    # print(result)
                total_sum += entry_sum
            f_alpha_expression_list.append(total_sum)
        return f_alpha_expression_list

    rep = len(datasets[0]) // m
    f_alpha_expressions = [
        process_dataset(dataset, lvals, solutions_combined) for dataset in datasets[:-1]
    ]
    y_values_dicts_list = solutions_combined

    ndim = datasets[0].shape[1]
    nlvals = len(lvals)
    f_vector = np.zeros((nsplits - 1, nlvals, len(datasets[-1])))
    variable_names = [f"x{i}" for i in range(1, ndim + 1)]

    for rowid in range(len(datasets[-1])):
        mapping = {var: datasets[-1][rowid, i] for i, var in enumerate(variable_names)}
        for split in range(nsplits - 1):
            for lidx in range(nlvals):
                f_vector[split, lidx, rowid] = f_alpha_expressions[split][lidx].evalf(
                    subs=mapping
                )

    # value1_0 = f_alpha_expressions[0][0].evalf(subs=row.to_dict())
    # value1_1 = f_alpha_expressions[0][1].evalf(subs=row.to_dict())

    # value1_2, value1_3 = eval(
    #    str(f_alpha_expression_1[2]), globals(), row.to_dict()
    # ), eval(str(f_alpha_expression_1[3]), globals(), row.to_dict())

    hs = [list(_.values()) for _ in y_values_dicts_list]
    RMSEs = np.zeros((nsplits - 1, len(lvals)))

    for split in range(nsplits - 1):
        for lidx in range(nlvals):
            RMSEs[split, lidx] = np.sqrt(
                mean_squared_error(np.repeat(hs[lidx], rep), f_vector[split, lidx, :])
            )

    h_value = hs[np.argmin(np.min(RMSEs, axis=0))]

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
                values = [
                    row[var] ** k if var != "1" else 1 for var, k in zip(variables, ks)
                ]
                term_value = coefficient * math.prod(values)
                expansions.append(term_value)
        return expansions[::-1]

    def multilinear_expansion1(variables1, n1):
        expansions1 = []
        for ks in itertools.product(range(n1 + 1), repeat=len(variables1)):
            if sum(ks) == n1:
                terms1 = [
                    f"{var}**{k}" if k != 0 else f"{var}"
                    for var, k in zip(variables1, ks)
                    if k != 0
                ]
                term1 = " * ".join(terms1)
                expansions1.append(term1)
        return expansions1[::-1]

    def generate_inner_products(coefficients, terms):
        inner_products = [f"{c}*{t}" for c, t in zip(coefficients, terms)]
        return inner_products


###########################################


df2 = pd.read_csv("50.csv")
df2["trajectory"] = df2["trajectory"].replace(
    {1: h_value[0], 2: h_value[1], 3: h_value[2], 4: h_value[3], 5: h_value[4]}
)
X = df2.iloc[:, :-1]
y = df2.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train.to_csv("X_train.csv", index=False)
y_train = y_train.astype(float)


def polynomial_kernel(X, Y, degree=1):
    return (1 + np.dot(X, Y.T)) ** degree


param_grid = {
    "alpha": [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
kr_model = KernelRidge(kernel=polynomial_kernel)
grid_search = GridSearchCV(
    kr_model, param_grid, cv=cv, scoring="neg_mean_squared_error"
)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best RMSE:", -grid_search.best_score_)
print("")
print("")
grid_search.best_estimator_
y_pred = grid_search.predict(X_test)


class KernelMethodBase(object):
    """
    Base class for kernel methods models
    Methods
    ----
    fit
    predict
    fit_K
    predict_K
    """

    kernels_ = {
        "polynomial": polynomial_kernel,
    }

    def __init__(self, kernel="polynomial", **kwargs):
        self.kernel_name = kernel
        self.kernel_function_ = self.kernels_[kernel]
        self.kernel_parameters = self.get_kernel_parameters(**kwargs)
        self.fit_intercept_ = False

    def get_kernel_parameters(self, **kwargs):
        params = {}
        params["degree"] = kwargs.get("degree", 1)
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
    """
    Kernel Ridge Regression
    """

    def __init__(self, alpha=0.1, **kwargs):
        self.alpha = alpha
        super(KernelRidgeRegression, self).__init__(**kwargs)

    def fit_K(self, K, y):
        n = K.shape[0]
        assert n == len(y)
        A = K + self.alpha * np.identity(n)
        self.eta = np.linalg.solve(A, y)
        return self

    def decision_function_K(self, K_x):
        return K_x.dot(self.eta)

    def predict(self, X):
        return self.decision_function(X)

    def predict_K(self, K_x):
        return self.decision_function_K(K_x)


kernel = "polynomial"
kr_model = KernelRidgeRegression(
    kernel=kernel,
    alpha=grid_search.best_params_["alpha"],
)
kr_model.fit(X_train, y_train)
x1, x2, x3, x4, x5, x6, x7, x8 = sp.symbols("x1 x2 x3 x4 x5 x6 x7 x8")
eta = kr_model.eta
num_variables = 8
data = pd.read_csv("X_train.csv")
variables = ["c"] + [f"x{i}" for i in range(1, num_variables + 1)]
n = 1
data["c"] = 1
variables1 = ["1"] + [f"x{i}" for i in range(1, num_variables + 1)]
n1 = 1
all_entries = []
for index, row in data.iterrows():
    result = multilinear_expansion(variables, n, row)
    result1 = multilinear_expansion1(variables1, n1)
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
filtered_terms = [
    term for coeff, term in zip(coefficients, terms) if abs(coeff) > 0.001
]
filtered = sum(
    sp.Mul(coeff, term)
    for coeff, term in zip(coefficients, terms)
    if term in filtered_terms
)
filtered_expression = sum(
    sp.Mul(coeff, term)
    for coeff, term in zip(coefficients, terms)
    if term in filtered_terms
)
print("Candidate Conservation Law:")
sp.pprint(total_sum)
print("")
print("")
with open("total_sum.txt", "w") as file:
    file.write(str(total_sum))
print("Final Candidate CL:")
sp.pprint(filtered_expression)
print("")
print("")
with open("total_sum.txt", "w") as file:
    file.write(str(total_sum))
with open("total_sum.txt", "r") as file:
    total_sum = sp.sympify(file.read())
df3 = pd.read_csv("holdoutp_data50.csv")
traj_len = df3.groupby("trajectory").size()
rep = int(round(traj_len.mean()))
expression = sp.lambdify((x1, x2, x3, x4, x5, x6, x7, x8), total_sum, "numpy")
df3["lamhold"] = expression(
    df3["x1"],
    df3["x2"],
    df3["x3"],
    df3["x4"],
    df3["x5"],
    df3["x6"],
    df3["x7"],
    df3["x8"],
)
da = {
    "y0": h_value[0],
    "y1": h_value[1],
    "y2": h_value[2],
    "y3": h_value[3],
    "y4": h_value[4],
}
df3["Coluh(lamhold)"] = [da[f"y{i}"] for i in range(5) for _ in range(rep)]
columns_to_compare = [("lamhold", "Coluh(lamhold)")]
for col1, col2 in columns_to_compare:
    rmse = np.sqrt(mean_squared_error(df3[col1], df3[col2]))
    print(f"Generalisation Error (RMSE): {rmse}")
    print("")
with open("total_sum.txt", "r") as file:
    total_sum = sp.sympify(file.read())
f = sp.lambdify((x1, x2, x3, x4, x5, x6, x7, x8), total_sum, "numpy")
dat = pd.read_csv("50.csv")
trajectories = dat["trajectory"].unique()
total_sum_squared_normalized_functional_value = 0
total_data_points = 0
for trajectory in trajectories:
    trajectory_data = dat[dat["trajectory"] == trajectory].copy()
    cols = ["x" + str(i) for i in range(1, 9)]  # number of variable
    trajectory_data["functional_value"] = f(*trajectory_data[cols].values.T)
    mean_value = trajectory_data["functional_value"].mean()
    trajectory_data["functional_value_minus_mean"] = (
        trajectory_data["functional_value"] - mean_value
    )
    trajectory_data["normalized_functional_value"] = (
        trajectory_data["functional_value_minus_mean"] / mean_value
    )
    trajectory_data["squared_normalized_functional_value"] = (
        trajectory_data["normalized_functional_value"] ** 2
    )
    total_sum_squared_normalized_functional_value += trajectory_data[
        "squared_normalized_functional_value"
    ].sum()
    total_data_points += len(trajectory_data)
average_squared_normalized_functional_value = (
    total_sum_squared_normalized_functional_value / total_data_points
)
standard_deviation = math.sqrt(average_squared_normalized_functional_value)
print(" Relative deviation:", standard_deviation)
print("")
df["x1"] = pd.to_numeric(df["x1"], errors="coerce")
df["x2"] = pd.to_numeric(df["x2"], errors="coerce")
df["x3"] = pd.to_numeric(df["x3"], errors="coerce")
df["x4"] = pd.to_numeric(df["x4"], errors="coerce")
df["x5"] = pd.to_numeric(df["x5"], errors="coerce")
df["x6"] = pd.to_numeric(df["x6"], errors="coerce")
df["x7"] = pd.to_numeric(df["x7"], errors="coerce")
df["x8"] = pd.to_numeric(df["x8"], errors="coerce")
f1_func = lambdify((x1, x2, x3, x4, x5, x6, x7, x8), filtered, "numpy")


def f2(x1, x2, x3, x4, x5, x6, x7, x8):
    return x1 + x2 + x3 + x4 - x5 - x6 - x7 - x8  # Exact Conservation law


dat["f1"] = f1_func(
    dat["x1"],
    dat["x2"],
    dat["x3"],
    dat["x4"],
    dat["x5"],
    dat["x6"],
    dat["x7"],
    dat["x8"],
)
dat["f2"] = dat.apply(
    lambda row: f2(
        row["x1"],
        row["x2"],
        row["x3"],
        row["x4"],
        row["x5"],
        row["x6"],
        row["x7"],
        row["x8"],
    ),
    axis=1,
)
corr, _ = pearsonr(dat["f1"], dat["f2"])
print(f"Correlation coefficient: {corr}")
plt.scatter(dat["f1"], dat["f2"], s=100, c=dat["trajectory"], cmap="viridis")
plt.xlabel("Learned", fontsize=18)
plt.ylabel("Exact", fontsize=18)
plt.xticks([])
plt.yticks([])
save_path = "Coerrelation_plot.pdf"
plt.savefig(save_path)
plt.close()
input_directory = r"."
output_directory = r"d"
df14 = pd.read_csv(
    os.path.join(input_directory, "50.csv")
)  # 50.csv, saved names of the data
Br = pd.read_csv("50.csv")
tr = Br.groupby("trajectory").size()
re1 = int(round(tr.mean()))
rows_per_file = re1
num_files = len(df14) // rows_per_file
data_chunks = [
    df14.iloc[i * rows_per_file : (i + 1) * rows_per_file] for i in range(num_files)
]
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
for i, chunk in enumerate(data_chunks):
    chunk.to_csv(os.path.join(output_directory, f"q{i + 1}.csv"), index=False)
expression = total_sum
with open("50.csv", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    current_trajectory = None
    expression_values = []
    for row in csv_reader:
        x1_val, x2_val, x3_val, x4_val, x5_val, x6_val, x7_val, x8_val, trajectory = (
            map(float, row)
        )
        if trajectory != current_trajectory:
            current_trajectory = trajectory
            initial_data = [
                x1_val,
                x2_val,
                x3_val,
                x4_val,
                x5_val,
                x6_val,
                x7_val,
                x8_val,
            ]
            expression_value = expression.subs(
                {
                    x1: x1_val,
                    x2: x2_val,
                    x3: x3_val,
                    x4: x4_val,
                    x5: x5_val,
                    x6: x6_val,
                    x7: x7_val,
                    x8: x8_val,
                }
            )
            expression_values.append(expression_value)
compute_function = sp.lambdify((x1, x2, x3, x4, x5, x6, x7, x8), total_sum, "numpy")
for i in range(1, 6):
    input_file = os.path.join(output_directory, f"q{i}.csv")
    output_file = os.path.join(output_directory, f"r{i}.csv")
    data = pd.read_csv(input_file)
    data["computed_value"] = compute_function(
        data["x1"],
        data["x2"],
        data["x3"],
        data["x4"],
        data["x5"],
        data["x6"],
        data["x7"],
        data["x8"],
    )
    data.to_csv(output_file, index=False)
subtraction_values = expression_values
for i in range(1, 6):  # 5 trajectory
    r_file = os.path.join(output_directory, f"r{i}.csv")
    n_file = os.path.join(output_directory, f"n{i}.csv")
    subtraction_value = subtraction_values[i - 1]
    data = pd.read_csv(r_file)
    data["adjusted_value"] = data["computed_value"] - subtraction_value
    data.to_csv(n_file, columns=["adjusted_value"], index=False)
all_data = pd.DataFrame()
subtraction_values = expression_values
for i in range(1, 6):  # 5 trajectory
    r_file = os.path.join(output_directory, f"r{i}.csv")
    n_file = os.path.join(output_directory, f"n{i}.csv")
    subtraction_value = subtraction_values[i - 1]
    data = pd.read_csv(r_file)
    data["adjusted_value"] = (
        data["computed_value"] - subtraction_value
    ) / subtraction_value
    data.to_csv(n_file, columns=["adjusted_value"], index=False)
    all_data[f"n{i}"] = data["adjusted_value"]
plt.figure(figsize=(10, 6))
for column in all_data.columns:
    plt.plot(all_data[column], label=column)
plt.xlabel("N", fontsize=16)
plt.ylabel("Relative Variation", fontsize=16)
save_path = "Relative_Variation.pdf"
plt.savefig(save_path)
plt.close()
