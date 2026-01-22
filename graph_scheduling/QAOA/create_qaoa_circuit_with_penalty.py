# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter
from typing import List


def find_best_pair(pairs: List, num_ops: List) -> List:
    """
    Given a list of pairs, and another list of the number of Rzz gates which have already been
    applied to each qubit, this function finds the pair of qubits [q1, q2] which have the lowest-
    maximum number of Rzz gates.

    E.g.
    pairs = [[0, 1], [2, 3]]
    num_ops = [1, 6, 3, 2]

    The maximum number of Rzz gates acting on either qubit in the pair [0, 1] is 6, and for the pair
    [2, 3] is 3, so [2, 3] is returned.

    Args:
        pairs (List): A list of pairs of qubits (each one a two-element list) on which Rzz gates
        need to be applied.
        num_ops (List): The number of Rzz gates which have already been applied to each qubit
        (ordered by qubit index)
    Returns:
        pair (List): The pair of qubits on which to add the next Rzz gate.
    """

    depths = [0] * len(pairs)
    for i in range(len(pairs)):
        pair = pairs[i]
        depth = max(num_ops[pair[0]], num_ops[pair[1]])
        depths[i] = depth

    pair = pairs[depths.index(min(depths))]

    return pair


def create_qaoa_circuit_with_penalty(
    hamiltonian: SparsePauliOp, num_qubits: int, p: int, reduce_depth: bool = True
) -> QuantumCircuit:
    """
    Constructs a QAOA circuit for a given Hamiltonian and specified parameters.

    Args:
        hamiltonian (SparsePauliOp): The problem Hamiltonian.
        num_qubits (int): Number of qubits.
        p (int): The number of layers to add to the circuit.
        reduce_depth (bool): Whether or not to prioritse parallel CNOT gates.

    Returns:
        QuantumCircuit
    """
    # Check consistency of inputs

    gammas = [Parameter(f"gamma_{i}") for i in range(p)]
    betas = [Parameter(f"beta_{i}") for i in range(p)]
    # Initialize the quantum circuit
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.h(range(num_qubits))  # Start with a uniform superposition

    # Define the QAOA layers
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        # Problem Hamiltonian layer
        pair_list = []
        coeff_list = []
        for term, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            # Pauli strings are interpreted in Qiskit's little-endian notation.
            # e.g., "IIIIZ" corresponds to Z acting on qubit 0
            pauli_string = term.to_label()
            z_indices = [
                (len(pauli_string) - i - 1)
                for i, pauli in enumerate(pauli_string)
                if pauli == "Z"
            ]

            if len(z_indices) == 1:
                # Linear term: Apply Rz to the single qubit
                circuit.rz(2 * coeff.real * gamma, z_indices[0])
            elif len(z_indices) == 2:
                if not reduce_depth:
                    # Quadratic term: Apply CNOTs and Rz
                    q1, q2 = z_indices
                    circuit.cx(q1, q2)
                    circuit.rz(2 * coeff.real * gamma, q2)
                    circuit.cx(q1, q2)
                else:
                    # Find all pairs of qubits on which to apply Rzz gates
                    pair_list.append(z_indices)
                    coeff_list.append(coeff)

        if reduce_depth:
            # Apply Rzz gates to the pairs in pair_list, prioritising low-depth.
            num_ops_list = [0] * circuit.num_qubits
            for _ in range(len(pair_list)):
                q1, q2 = find_best_pair(pair_list, num_ops_list)
                coeff = coeff_list[pair_list.index([q1, q2])]
                circuit.cx(q1, q2)
                circuit.rz(2 * coeff.real * gamma, q2)
                circuit.cx(q1, q2)
                num_ops_list[q1] += 1
                num_ops_list[q2] += 1
                pair_list.remove([q1, q2])
                coeff_list.remove(coeff)

        # Mixer Hamiltonian layer
        for qubit in range(num_qubits):
            circuit.rx(2 * beta, qubit)
    circuit.measure(range(num_qubits), range(num_qubits))

    return circuit
