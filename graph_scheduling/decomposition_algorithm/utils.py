# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import networkx as nx
import numpy as np
from qiskit_ibm_runtime import Session
from typing import Tuple, List, Dict
from graph_scheduling.QAOA.iterative_qaoa_optimization import (
    compute_qaoa_matching,
    iterative_qaoa_optimization,
    evaluate_hamiltonian_value,
)
from graph_scheduling.QAOA.create_qaoa_circuit_with_penalty import (
    create_qaoa_circuit_with_penalty,
)
from graph_scheduling.QAOA.utils import (
    construct_qubo,
    generate_cost_hamiltonian,
    solve_qubo_sa,
    generate_random_samp_dist,
    bitstring_to_matching_with_mapping,
    is_valid_matching,
    solve_with_networkx,
)


# -----------------------------------------------
# --- Matching-manipulation utility functions ---
# -----------------------------------------------


def matching_to_matrix(matching: List[List] | List[Tuple], n: int) -> np.ndarray:
    """
    Given a matching of the form:
        [(node_1, node_2), ...] or [[node_1, node_2], ...],
    return an n x n matrix containing 1 at elements indexed by all (node_i, node_j) (and symmetric)
    for all edges in the matching.

    Args:
        matching: The list of edges in the matching.
        n: the size of the matrix to return.
    Returns
        M: The matrix representation of the matching.
    """
    M = np.zeros((n, n))
    for edge in matching:
        M[edge[0], edge[1]] = 1
        M[edge[1], edge[0]] = 1

    return M


def find_unique_matchings_and_weights(matchings: List, weights: List) -> Tuple[List]:
    """
    Given a list of matchings and a list of weights, this function removes any duplicated matchings,
    and adds together the corresponding weights. For example:

    matchings = [A, B, C, A]  -> new_matchings = [A, B, C]
    weights = [1, 2, 3, 4]       new_weights = [5, 2, 3]

    Args:
        matchings (List): The list of matchings.
        weights (List): The list of weights.
    Returns:
        unique_matchings (List): A list corresponging to `matchings` with any duplicates removed.
        unique_weights (List): A list containing the cumulative weight for each matching.
    """
    unique_matchings = [matchings[0]]
    unique_weights = [weights[0]]
    for i in range(1, len(matchings)):
        unique = True
        for j in range(len(unique_matchings)):
            if np.array_equal(matchings[i], unique_matchings[j]):
                unique_weights[j] += weights[i]
                unique = False
                break

        if unique:
            unique_matchings.append(matchings[i])
            unique_weights.append(weights[i])

    return unique_matchings, unique_weights


def check_matching(M: np.ndarray, A: np.ndarray) -> bool:
    """
    Check that M is a matching in A:
        1. For any A[i, j] == 0, must have M[i, j] == 0
        2. The sum of any row or col in M must be 0 or 1

    Args:
        M (np.ndarray): The prospective matching.
        A (np.ndarray): The matrix to check that M is a matching of.
    Returns:
        True if M is a matching in A, False otherwise.
    """
    if np.any(M[A == 0] != 0):
        return False

    row_sums = M.sum(1)
    col_sums = M.sum(0)

    if np.any(row_sums > 1) or np.any(col_sums > 1):
        return False
    else:
        return True


def filter_out_non_matchings(old_dict, edge_to_qubit):
    """
    Given a dictionary {bitstrings: values}, return a new dictionary only containing entries where
    the bitstrings correspond to valid matchings.
    """
    new_dict = {}
    for key, value in old_dict.items():
        if is_valid_matching(bitstring_to_matching_with_mapping(key, edge_to_qubit)):
            new_dict[key] = value

    return new_dict


# ------------------------------------------
# --- Matching-finding utility functions ---
# ------------------------------------------


def nx_max_weight_matching(matrix: np.ndarray) -> np.ndarray:
    """
    Given an adjacency matrix, find the maximally-weighted matching using NetworkX.

    Args:
        matrix (np.ndarray): The matrix for which to find the maximally-weighted matching.
    Returns:
        M (np.ndarray): The maximally-weighted matching.
    """

    G = nx.from_numpy_array(matrix, create_using=nx.Graph)
    # NOTE: Setting maxcardinality=True will find the maximally-weighted maximum cardinality
    # matching.
    # This is not guaranteed to have the absolute maximum cardinality, since it is subject
    # to constraints.
    # E.g. if there are 16 nodes, the maximum possible cardinality is 8, but depending on
    # how many 0s are in the matrix, the maximum cardinality of valid matchings may be
    # less than 8
    maximal_matching = list(nx.max_weight_matching(G, maxcardinality=True))
    # Turn matching into a matrix
    M = matching_to_matrix(maximal_matching, matrix.shape[0])

    assert check_matching(M, matrix)

    return M


def sa_high_weight_matchings(
    matrix: np.ndarray, num_matchings: int = 1
) -> List[np.ndarray]:
    """
    Given an adjacency matrix, find the n lowest energy matchings using simulated annealing.

    Args:
        matrix (np.ndarray): The matrix for which to find the maximally-weighted matchings.
        mum_matchings (int): The number of matchings to return.
    Returns:
        matchings (List[np.ndarray]): The `num_matchings` highest-weight matchings.
    """

    G = nx.from_numpy_array(matrix, create_using=nx.Graph)
    edges = list(G.edges())
    edge_to_qubit = {edge: idx for idx, edge in enumerate(edges)}
    print(edge_to_qubit)
    qubo, _, _ = construct_qubo(G, edge_to_qubit, penalty_scale=0.2)

    # TODO: How many samples should we take?
    _, _, solutions, energies = solve_qubo_sa(qubo, num_reads=1000)

    # Get SA solutions into the standard matching form
    sa_matchings = [
        [edge for edge, value in solution.items() if value == 1]
        for solution in solutions
    ]

    # SA solutions are not necessarily unique, and are not necessarily valid matchings.
    # Here we extract the unique and valid matchings.
    unique_valid_matchings = []
    unique_energies = []

    for matching, energy in zip(sa_matchings, energies):
        if matching not in unique_valid_matchings and is_valid_matching(matching):
            unique_valid_matchings.append(matching)
            unique_energies.append(energy)

    matchings = []
    for matching in unique_valid_matchings:
        if len(matchings) < num_matchings:
            print(matching)
            M = matching_to_matrix(matching, matrix.shape[0])

            matchings.append(M)

    return matchings


def qaoa_high_weight_matchings(
    matrix: np.ndarray,
    num_matchings: int = 1,
    p: int = 1,
    train_params: bool = True,
    params: Dict = None,
    session: Session = None,
) -> List[np.ndarray]:
    """
    Given an adjacency matrix, find the n lowest energy matchings using QAOA.

    Args:
        matrix (np.ndarray): The matrix for which to find the maximally-weighted matchings.
        mum_matchings (int): The number of matchings to return.
        p (int): The number of layers to use for the QAOA ansatz.
        train_params (bool): Whether or not to train the parameters in the QAOA ansatz.
        params (Dict): Dictionary containing beta and gammma values, of the form:
            params = {
                p0: {"beta": ..., "gamma": ...},
                ...,
                p_final: {"beta": ..., "gamma": ...},
            }
            The beta and gamma entries must be np.ndarrays of length equal to the given value of p.
            If training==False, these parameters will be used as specified. If training==True, only
            the p=1 parameters will be used as initialisation.
        session (Session): If a session is provided, circuits will be executed on hardware.
    Returns:
        matchings (List[np.ndarray]): The `num_matchings` highest-weight matchings.
    """

    G = nx.from_numpy_array(matrix, create_using=nx.Graph)
    edges = list(G.edges())
    edge_to_qubit = {edge: idx for idx, edge in enumerate(edges)}
    print(edge_to_qubit)
    _, _, qubo_edge = construct_qubo(G, edge_to_qubit, penalty_scale=0.2)
    hamiltonian, _ = generate_cost_hamiltonian(qubo_edge)

    matching, _, _, _, _, _, _, _ = compute_qaoa_matching(
        hamiltonian=hamiltonian,
        edge_to_qubit=edge_to_qubit,
        p_final=p,
        steps=p,
        iterative_qaoa_optimization_fn=iterative_qaoa_optimization,
        qaoa_circuit_fn=create_qaoa_circuit_with_penalty,
        num_valid_matchings=num_matchings,
        nx_weight=solve_with_networkx(G)[1],
        weights=nx.get_edge_attributes(G, "weight"),
        use_estimator=False,
        QCTRL=False,
        training=train_params,
        session=session,
        params=params,
    )
    qaoa_matchings = matching[p]

    matchings = []
    for matching in qaoa_matchings:
        print(matching)
        M = matching_to_matrix(matching, matrix.shape[0])

        if check_matching(M, matrix):
            matchings.append(M)
        else:
            print("Warning: skipping as QAOA result is not a matching")

    return matchings


def random_high_weight_matchings(
    matrix: np.ndarray, num_matchings: int = 1
) -> List[np.ndarray]:
    """
    Given an adjacency matrix, find the n lowest energy matchings using random sampling, rather than
    QAOA.

    Args:
        matrix (np.ndarray): The matrix for which to find the maximally-weighted matchings.
        mum_matchings (int): The number of matchings to return.
    Returns:
        matchings (List[np.ndarray]): The `num_matchings` highest-weight matchings.
    """

    G = nx.from_numpy_array(matrix, create_using=nx.Graph)
    edges = list(G.edges())
    edge_to_qubit = {edge: idx for idx, edge in enumerate(edges)}
    print(edge_to_qubit)
    _, _, qubo_edge = construct_qubo(G, edge_to_qubit, penalty_scale=0.2)
    hamiltonian, _ = generate_cost_hamiltonian(qubo_edge)

    samp_dist = generate_random_samp_dist(hamiltonian.num_qubits, int(1e4))

    samp_dist = filter_out_non_matchings(samp_dist, edge_to_qubit)

    energies = {
        state: (evaluate_hamiltonian_value(hamiltonian, state))
        for state, count in samp_dist.items()
    }

    sorted_bitstrings = sorted(energies, key=energies.get)
    best_n_bitstrings = sorted_bitstrings[:num_matchings]

    random_matchings = [
        bitstring_to_matching_with_mapping(bitstring, edge_to_qubit)
        for bitstring in best_n_bitstrings
    ]

    matchings = []
    for matching in random_matchings:
        print(matching)
        M = matching_to_matrix(matching, matrix.shape[0])

        if check_matching(M, matrix):
            matchings.append(M)
        else:
            print("Warning: skipping as QAOA result is not a matching")

    return matchings


# ---------------------------------------
# --- Miscellaneous utility functions ---
# ---------------------------------------


def get_current_demand(
    demand_matrix: np.ndarray, coeffs: List[float], matchings: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a demand matrix, list of coefficients, and list of matchings, calculate the 'current
    demand', defined as:

    current_demand = demand_matrix - sum_i(coeffs[i] * matchings[i]),

    Args:
        demand_matrix (np.ndarray): The demand matrix.
        coeffs (List[float]): The current weights in the decomposition.
        matchings (List[np.ndarray]): The current matchings in the decomposition.
    Returns:
        current_demand (np.ndarray): The current_demand, as defined above.
    """

    current_decomp = np.zeros(demand_matrix.shape)
    for i in range(len(coeffs)):
        current_decomp += coeffs[i] * matchings[i]
    current_demand = demand_matrix - current_decomp

    return current_demand
