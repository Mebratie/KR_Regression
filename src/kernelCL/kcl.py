import numpy as np
import pandas as pd
import sympy as sp
import itertools
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
)
from sympy import lambdify
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
import math
import sys


def combine_splits(indices, datasets):
    combined_data = np.vstack([datasets[i] for i in indices])
    return combined_data


if __name__ == "__main__":
    datafiles = sys.argv[1:]
    if len(datafiles) == 0:
        print("No trajectory data provided.")
        sys.exit(1)

    # configure here
    lvals = [10 ** (-i) for i in range(8)]
    c, d = 1, 1  # Adjust the values of variable "d" according to the required degree.

    m = len(datafiles)
    nsplits = 5
    training_frac = 0.8

    # load trajectories
    print("Loading trajectories")
    trajs = []
    for datafile in datafiles:
        traj = np.loadtxt(datafile, delimiter=",", skiprows=1)
        trajs.append(traj)
    num_variables = traj.shape[1]
    column_names_x = [f"x{_}" for _ in range(1, num_variables + 1)]
    column_names_y = [f"y{_}" for _ in range(len(trajs))]

    # split data into training and holdout sets
    datasets = [list() for _ in range(nsplits)]
    holdout = []
    training = []
    full = []
    for traj in trajs:
        npts = int(traj.shape[0] * training_frac) // nsplits
        np.random.shuffle(traj)
        full.append(traj)

        for split in range(nsplits):
            part = traj[split * npts : (split + 1) * npts, :]
            datasets[split].append(part)
        ntrain = int(traj.shape[0] * training_frac)
        training.append(traj[:ntrain, :])
        holdout.append(traj[ntrain:, :])
    datasets = [np.vstack(ds) for ds in datasets]
    #############################################
    combined_datasets = []
    for i in range(nsplits):
        indices = [j for j in range(nsplits) if j != i]
        combined_data = combine_splits(indices, datasets)
        combined_datasets.append(combined_data)

    #################################################
    def compute_kernel_matrix(X, c, d):
        K = (c + np.dot(X, X.T)) ** d
        return K

    def solve_for_lambda(data, c, d, lambda_value, m):
        rep = len(data) // m
        K = compute_kernel_matrix(data, c, d)
        I = np.eye(K.shape[0])
        K_with_I = K + lambda_value * I
        K_with_I_inv = np.linalg.inv(K_with_I)
        y0 = sp.symbols("y0")
        y_pattern = [sp.symbols(f"y{i}") for i in range(m)]
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

    def add_matrices(mat1, mat2, mat3, mat4, mat5):
        return mat1 + mat2 + mat3 + mat4 + mat5

    print("Determine regularisation")
    solutions = [
        [solve_for_lambda(dataset, c, d, l, m) for l in lvals] for dataset in datasets
    ]
    solutions_combined = []
    for i, (w1, w2, w3, w4, w5) in enumerate(zip(*solutions)):
        added_matrix = add_matrices(w1["W"], w2["W"], w3["W"], w4["W"], w5["W"])
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

    def process_dataset(combined_dataset, lambda_values, y_values_dicts):
        data = pd.DataFrame(combined_dataset, columns=column_names_x)
        variables = ["c"] + column_names_x
        n = 1  # degree of the polynomial kernel
        data["c"] = 1
        variables1 = ["1"] + column_names_x
        n1 = 1  # degree of the polynomial kernel
        all_entries = []
        f_alpha_expression_list = []
        data_np = combined_dataset
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
            for entry_index, (entry, alpha) in enumerate(
                zip(all_entries, alpha_sym), 1
            ):
                entry_sum = 0
                for term in entry:
                    result = alpha * sp.sympify(term)
                    entry_sum += result
                total_sum += entry_sum
            f_alpha_expression_list.append(total_sum)
        return f_alpha_expression_list

    rep = len(combined_datasets[0]) // m
    f_alpha_expressions = [
        process_dataset(combined_dataset, lvals, solutions_combined)
        for combined_dataset in combined_datasets
    ]
    y_values_dicts_list = solutions_combined

    ndim = num_variables
    nlvals = len(lvals)
    nrows = len(combined_datasets[-1])
    f_vector = np.zeros((nsplits, nlvals, nrows))
    variable_names = [f"x{i}" for i in range(1, ndim + 1)]

    for rowid in range(nrows):
        for split in range(nsplits):
            mapping = {
                var: combined_datasets[split][rowid, i]
                for i, var in enumerate(variable_names)
            }
            for lidx in range(nlvals):
                f_vector[split, lidx, rowid] = f_alpha_expressions[split][lidx].evalf(
                    subs=mapping
                )

    hs = [list(_.values()) for _ in y_values_dicts_list]
    RMSEs = np.zeros((nsplits, len(lvals)))

    for split in range(nsplits):
        for lidx in range(nlvals):
            RMSEs[split, lidx] = np.sqrt(
                mean_squared_error(np.repeat(hs[lidx], rep), f_vector[split, lidx, :])
            )
    h_value = hs[np.argmin(np.average(RMSEs, axis=0))]

    def generate_inner_products(coefficients, terms):
        inner_products = [f"{c}*{t}" for c, t in zip(coefficients, terms)]
        return inner_products

    X_train = np.vstack(training)
    y_train = np.hstack(
        [np.repeat(h_value[_], len(training[_])) for _ in range(len(training))]
    ).astype(float)

    def polynomial_kernel(X, Y, degree=1):
        return (1 + np.dot(X, Y.T)) ** degree

    class KernelMethodBase(BaseEstimator):
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
            K = self.kernel_function_(
                self.X_train, self.X_train, **self.kernel_parameters
            )
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

    print("Fit overall model")
    kernel = "polynomial"
    param_grid = {
        "alpha": [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    kr_model = KernelRidgeRegression(kernel=kernel)
    grid_search = GridSearchCV(
        kr_model, param_grid, cv=cv, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best RMSE:", -grid_search.best_score_)

    kr_model = KernelRidgeRegression(
        kernel=kernel,
        alpha=grid_search.best_params_["alpha"],
    )
    kr_model.fit(X_train, y_train)

    column_symbols = sp.symbols(" ".join(column_names_x))
    eta = kr_model.eta
    data = pd.DataFrame(X_train, columns=column_names_x)
    variables = ["c"] + column_names_x
    n = 1
    data["c"] = 1
    variables1 = ["1"] + column_names_x
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

    nholdouts = sum([len(_) for _ in holdout])
    holdout_data = np.zeros((nholdouts, num_variables + 1))
    holdout_data[:, :-1] = np.vstack(holdout)
    holdout_data[:, -1] = np.hstack(
        [np.repeat(_, len(holdout[_])) for _ in range(len(holdout))]
    ).astype(float)
    df3 = pd.DataFrame(
        holdout_data,
        columns=[f"x{i}" for i in range(1, num_variables + 1)] + ["trajectory"],
    )

    traj_len = df3.groupby("trajectory").size()
    rep = int(round(traj_len.mean()))
    expression = sp.lambdify(column_symbols, total_sum, "numpy")
    df3["lamhold"] = expression(*[df3[_] for _ in column_names_x])

    da = {column_names_y[i]: h_value[i] for i in range(len(trajs))}
    df3["Coluh(lamhold)"] = [da[f"y{i}"] for i in range(len(trajs)) for _ in range(rep)]
    columns_to_compare = [("lamhold", "Coluh(lamhold)")]
    for col1, col2 in columns_to_compare:
        rmse = np.sqrt(mean_squared_error(df3[col1], df3[col2]))
        print(f"Generalisation Error (RMSE): {rmse}")
        print("")
    with open("total_sum.txt", "r") as file:
        total_sum = sp.sympify(file.read())
    f = sp.lambdify(column_symbols, total_sum, "numpy")

    nfulls = sum([len(_) for _ in full])
    full_data = np.zeros((nfulls, num_variables + 1))
    full_data[:, :-1] = np.vstack(full)
    full_data[:, -1] = np.hstack(
        [np.repeat(_, len(full[_])) for _ in range(len(full))]
    ).astype(float)
    dat = pd.DataFrame(
        full_data,
        columns=column_names_x + ["trajectory"],
    )

    trajectories = dat["trajectory"].unique()
    total_sum_squared_normalized_functional_value = 0
    total_data_points = 0
    for trajectory in trajectories:
        trajectory_data = dat[dat["trajectory"] == trajectory].copy()
        trajectory_data["functional_value"] = f(
            *trajectory_data[column_names_x].values.T
        )
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
    df = pd.DataFrame(datasets[-1], columns=column_names_x)
    for col in column_names_x:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    f1_func = lambdify(column_symbols, filtered, "numpy")
