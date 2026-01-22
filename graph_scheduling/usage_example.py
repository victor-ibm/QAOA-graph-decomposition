# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import json
import numpy as np
from graph_scheduling.decomposition_algorithm.decompose import Decomposer

# Load a demand matrix from the complete-graph dataset
n = 6
id = "6"

fn = f"./dataset/complete/graph_{n}_sparse.json"

with open(fn) as file:
    data = json.load(file)

D = data[id]["scaled_matrix"]
scale = data[id]["scale"]
demand_matrix = np.array(D).reshape((n, n)) / scale

# Initialise the decomposer
decomposer = Decomposer(
    demand_matrix=demand_matrix,
    tol=1e-6,
    method="E-FCFW",
    renormalise_weights=True,
    e_fcfw_config={
        "matchings_per_iteration": 5,
        "sampling_method": "QAOA",
        "qaoa_config": {
            "num_qaoa_layers": 1,
            "train_params": True,
            "simulation_method": "simulator",
        },
    },
)

# Decompose
result = decomposer.decompose()

coeffs = result["weights"]
matchings = result["matchings"]
cost_history = result["cost_history"]

result = np.zeros((n, n))
for i in range(len(coeffs)):
    result += coeffs[i] * matchings[i]

print("Demand matrix:")
print(demand_matrix)
print("Reconstructed matrix:")
print(result)
print(f"Decomposition length: {len(coeffs)}")
print(f"Coeffs: {coeffs}")
print(f"Decomposition error: {cost_history[-1]}")
