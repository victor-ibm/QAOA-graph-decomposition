# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import networkx as nx
import random
from qiskit_optimization import QuadraticProgram
import dimod
from sympy import symbols
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from typing import List, Tuple


def generate_random_graph(n, p):
    """
    n:number of nodes
    p:probability
    """
    graph = nx.gnp_random_graph(n, p, seed=random.randint(0, 100))
    for u, v in graph.edges():
        graph[u][v]["weight"] = random.randint(1, 10)
    return graph


def generate_cost_hamiltonian(qubo_edge):
    quadratic_coeffs = {
        (str(var1), str(var2)): coeff
        for (var1, var2), coeff in qubo_edge.quadratic.items()
    }
    linear_coeffs = {str(var): coeff for var, coeff in qubo_edge.linear.items()}
    qp = QuadraticProgram()
    for var in list(qubo_edge.variables):
        qp.binary_var(var)
    qp.minimize(linear=linear_coeffs, quadratic=quadratic_coeffs)
    operator, offset = qp.to_ising()

    # Normalise the Hamiltonian
    max_weight = max(abs(operator.coeffs))
    normalised_operator = SparsePauliOp(operator.paulis, operator.coeffs / max_weight)

    return normalised_operator, offset


def sort_matching(matching):
    # Sort nodes in each edge and then sort all edges lexicographically
    return sorted(tuple(sorted(edge)) for edge in matching)


def construct_qubo(graph, edge_to_qubit, penalty_scale=1):

    lambda_param = penalty_scale * sum(nx.get_edge_attributes(graph, "weight").values())
    qubo = dimod.BinaryQuadraticModel(dimod.Vartype.BINARY)
    qubo_edge = dimod.BinaryQuadraticModel(dimod.Vartype.BINARY)
    variables = {}

    # Define symbolic variables for edges
    edge_vars = {}
    for idx, edge in enumerate(graph.edges()):
        edge_vars[edge] = symbols(f"x_{edge_to_qubit[edge]}")
        variables[edge] = idx

    # Add linear terms to the QUBO
    for u, v, w_uv in graph.edges(data="weight"):
        qubo.add_linear((u, v), -w_uv)
        qubo_edge.add_linear(f"x_{edge_to_qubit[(u,v)]}", -w_uv)

    # Add quadratic terms to the QUBO
    count = 0
    for u, v in graph.edges():
        for u_prime, v_prime in graph.edges():
            if (u, v) != (u_prime, v_prime) and (
                u in (u_prime, v_prime) or v in (u_prime, v_prime)
            ):
                if edge_to_qubit[(u, v)] > edge_to_qubit[u_prime, v_prime]:
                    continue
                count += 1
                qubo.add_quadratic((u, v), (u_prime, v_prime), lambda_param)
                qubo_edge.add_quadratic(
                    f"x_{edge_to_qubit[(u,v)]}",
                    f"x_{edge_to_qubit[u_prime, v_prime]}",
                    lambda_param,
                )

    return qubo, edge_vars, qubo_edge


def solve_qubo(qubo):
    sampler = dimod.ExactSolver()
    response = sampler.sample(qubo)
    best_solution = response.first.sample
    matching = [edge for edge, value in best_solution.items() if value == 1]
    return sort_matching(matching), response.first.energy


def solve_qubo_sa(qubo, num_reads=1000):
    """
    Solve the QUBO using simulated annealing.
    """
    sampler = dimod.SimulatedAnnealingSampler()
    response = sampler.sample(qubo, num_reads=num_reads)

    # Extract all solutions and their energies
    solutions = [sample for sample in response.samples(sorted_by="energy")]
    energies = [sample.energy for sample in response.data(sorted_by="energy")]

    # Extract the best solution
    best_solution = response.first.sample
    matching = [edge for edge, value in best_solution.items() if value == 1]

    return sort_matching(matching), response.first.energy, solutions, energies


def solve_with_networkx(graph):
    matching = nx.max_weight_matching(graph, maxcardinality=False)
    weight = sum(graph[u][v]["weight"] for u, v in matching)
    return sort_matching(matching), weight


def compute_ground_state(hamiltonian):
    eigensolver = NumPyMinimumEigensolver()
    result = eigensolver.compute_minimum_eigenvalue(hamiltonian)
    ground_energy = result.eigenvalue.real
    ground_state = result.eigenstate
    return ground_energy, ground_state


# Function to interpret ground state
def interpret_ground_state(ground_state, edge_to_qubit):
    probabilities = np.abs(ground_state) ** 2
    dominant_index = np.argmax(probabilities)
    num_qubits = len(edge_to_qubit)
    binary_solution = f"{dominant_index:0{num_qubits}b}"
    print(" binary_solution", binary_solution)
    selected_edges = [
        edge
        for edge, qubit in edge_to_qubit.items()
        if binary_solution[num_qubits - qubit - 1] == "1"
    ]
    return sort_matching(selected_edges)


def bitstring_to_matching_with_mapping(bitstring, edge_to_qubit):
    """
    Translates a QAOA output bitstring into a matching based on an edge-to-qubit mapping.

    Args:
        bitstring (str): The bitstring output from QAOA.
        edge_to_qubit (dict): A dictionary mapping edges (tuples) to qubit indices.

    Returns:
        list of tuples: The edges included in the matching.
    """
    matching = []
    for edge, qubit in edge_to_qubit.items():
        if (
            bitstring[len(bitstring) - qubit - 1] == "1"
        ):  # If the corresponding qubit is 1, include the edge
            matching.append(edge)
    return sort_matching(matching)


def calculate_matching_weight(matching: List[Tuple], weights: dict) -> float:
    """
    Given a matching and dictionary of edge weights, return the weight of the matching.

    Args:
        matching (List[Tuple]): The edges in the matching.
        weights (dict): The weights of the edges in the graph.

    Returns:
        The weight of the matching
    """

    return sum([weights[edge] for edge in matching])


def is_valid_matching(edges: List[Tuple]) -> bool:
    """
    Check if a list of edges is a valid matching.

    Args:
        edges (List[Tuple]): The edges to check.
    Returns:
        True if no edges share a common node, False otherwise.
    """

    nodes_in_matching = set()  # Set to track nodes already used in the matching
    for u, v in edges:
        if u in nodes_in_matching or v in nodes_in_matching:
            return False  # If any node is reused, it's not a valid matching
        nodes_in_matching.add(u)
        nodes_in_matching.add(v)

    return True


def generate_random_samp_dist(num_bits: int, num_samples: int) -> dict:
    """
    Generates `num_samples` uniformly random bitstrings of length `num_bits`.

    Args:
        num_bits (int): The number of bits in the bitstring.
        num_samples (int): The number of samples to take.
    Returns:
        samp_dist (dict): A dictionary of the form {'bitstring': count}.
    """

    samp_dist = {}

    for i in range(num_samples):
        rand_bitstring_list = [str(np.random.randint(0, 2)) for i in range(num_bits)]
        rand_bitstring = "".join(rand_bitstring_list)
        if rand_bitstring in samp_dist.keys():
            samp_dist[rand_bitstring] += 1
        else:
            samp_dist[rand_bitstring] = 1

    return samp_dist
