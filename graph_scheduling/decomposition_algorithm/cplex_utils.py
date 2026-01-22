# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from docplex.mp.model import Model
from docplex.mp.dvar import Var
from typing import Tuple, List


def matrix(x: Var, n: int) -> np.ndarray:
    """
    Construct an nxn identity matrix scaled by the variable x: x * Id_n
    For some reason x * np.ndarray throws a dimension error, when float * np.ndarray does not.

    Args:
        x (docplex.mp.dvar.Var): The variable.
        n (int): the size of the matrix.
    Returns:
        x * Id_n (np.ndarray).
    """

    matrix = np.eye(n, dtype=object)
    for i in range(n):
        matrix[i, i] = x

    return matrix


def cplex_minimise_cost(
    D: np.ndarray, M: List[np.ndarray], max_length: int = None
) -> Tuple[List[float], float]:
    """
    Given a demand matrix D and a list of matchings M, find x to minimise the cost:
    ||D - x.M||^2_F / n^2     (Frobenius norm squared, normalised by the number of nodes squared),
    whilst keeping the number of non-zero elements of x less than or equal to max_length

    Args:
        D (np.ndarray): The demand matrix.
        M (List[np.ndarray]): The list of matchings.
        max_length (int): The maximum number of non-zero variables allowed (if None, no limit)
    Returns:
        x_opt (List[float]): The optimal parameters.
        cost (float): The minimum cost.
    """

    mdl = Model()
    x = mdl.continuous_var_list(keys=len(M), lb=0, ub=1)
    mdl.add_constraint(sum(x) <= 1)

    # Minimise ||D - x.M||^2_F.
    X = sum([matrix(x[i], D.shape[0]) @ M[i] for i in range(len(M))])
    norm_squared = np.sum(np.square(D - X))
    mdl.minimize(norm_squared / (D.shape[0] ** 2))

    if max_length is not None:
        # Define binary variables x_on, such that if x_on[i]==0, then x[i]==0.
        # The sum of x_on gives us the number of non-zero weights.
        x_on = mdl.binary_var_list(keys=len(M))
        for i in range(len(M)):
            mdl.add_if_then(x_on[i] == 1, x[i] >= 0)
            mdl.add_if_then(x_on[i] == 0, x[i] == 0)

        num_non_zero = sum(x_on)

        # Constraint to keep the length below a certain value.
        mdl.add(num_non_zero <= max_length)

    solution = mdl.solve()

    x_opt = solution.get_values(x)
    cost = solution.objective_value

    return x_opt, cost


def cplex_minimise_length(
    D: np.ndarray, M: List[np.ndarray], tol: float
) -> List[float]:
    """
    Given a demand matrix D, a list of matchings M, and a tolerance tol, find the minimum possible
    number of non-zero parameters x, such that the cost:
    ||D - x.M||^2_F / n^2     (Frobenius norm squared, normalised by the number of nodes squared),
    is below the tolerance.

    NOTE: This is a single-objective minimisation. It will find the minimal number of parameters
    such that the cost is below tol. It will not try to minimise the cost further.

    Args:
        D (np.ndarray): The demand matrix.
        M (List[np.ndarray]): The list of matchings.
        tol (float): The tolerance.
    Returns:
        x_opt (List[float]): The optimal parameters.
    """
    mdl = Model()
    x = mdl.continuous_var_list(keys=len(M), lb=0, ub=1)
    mdl.add_constraint(sum(x) <= 1)

    # norm_squared = ||D - x.M||^2_F
    X = sum([matrix(x[i], D.shape[0]) @ M[i] for i in range(len(M))])
    norm_squared = np.sum(np.square(D - X))

    # Define binary variables x_on, such that if x_on[i]==0, then x[i]==0.
    # The sum of x_on gives us the number of non-zero weights.
    x_on = mdl.binary_var_list(keys=len(M))
    for i in range(len(M)):
        mdl.add_if_then(x_on[i] == 1, x[i] >= 0)
        mdl.add_if_then(x_on[i] == 0, x[i] == 0)

    num_non_zero = sum(x_on)

    # Minimise the number of non-zero weights whilst keeping cost <= tol.
    mdl.add(norm_squared / (D.shape[0] ** 2) <= tol)
    mdl.minimize(num_non_zero)

    solution = mdl.solve()

    x_opt = solution.get_values(x)

    return x_opt
