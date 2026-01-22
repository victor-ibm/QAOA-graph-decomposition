# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from graph_scheduling.QAOA.parameter_initialization import interpolate_initialize
from qiskit_ibm_runtime import Session, SamplerOptions, QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.quantum_info import SparsePauliOp
from graph_scheduling.QAOA.utils import bitstring_to_matching_with_mapping
from collections.abc import Callable
from typing import List, Tuple, Dict
import warnings
from graph_scheduling.QAOA.utils import calculate_matching_weight


from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_catalog import QiskitFunctionsCatalog

warnings.simplefilter("ignore", np.exceptions.ComplexWarning)


def iterative_qaoa_optimization(
    qaoa_circuit_fn: Callable,
    hamiltonian: SparsePauliOp,
    p_final: int,
    steps: int,
    use_estimator: bool,
    training: bool,
    QCTRL: bool = False,
    session: Session = None,
    params: Dict = None,
) -> Tuple[dict]:
    """
    Iteratively optimize QAOA parameters for increasing circuit depths.

    Parameters:
        qaoa_circuit_fn (Callable): QAOA circuit template.
        hamiltonian (SparsePauliOp): Problem Hamiltonian.
        p_final (int): Maximum depth for QAOA.
        steps (int): Number of steps from 1 to p_final.
        use_estimator (bool): Use estimator(sampler) for training if True(False).
        QCTRL (bool): Use QCTRL for execution if True.
        training (bool): Whether or not to train the QAOA parameters.
        session (Session): If a session is provided, circuits will be executed on hardware.
            NOTE: You can run the jobs in "job" mode by passing a Backend as the `session` argument.
        params (Dict): Dictionary containing beta and gammma values, of the form:
            params = {
                p0: {"beta": ..., "gamma": ...},
                ...,
                p_final: {"beta": ..., "gamma": ...},
            }
            The beta and gamma entries must be np.ndarrays of length equal to the given value of p.
            If training==False, these parameters will be used as specified. If training==True, only
            the p=1 parameters will be used as initialisation.

    Returns:
        results (dict): Optimized parameters for each depth.
        qaoa_circuits (dict): QAOA circuits for each depth.
    """
    # If no initial parameters are specified, set default.
    if params is None:
        if not training:
            raise ValueError(
                "You must specify beta and gamma parameters if not training."
            )
        print("Warning: using default beta and gamma values (0.6) for initialisation.")
        params = {1: {"beta": np.array([0.6]), "gamma": np.array([0.6])}}

    hardware = True if session is not None else False

    # Initialize results dictionary
    results = {}
    qaoa_circuits = {}
    transpiled_qaoa_circuits = {}
    p_values = np.linspace(1, p_final, steps, dtype=int)

    # Initial values for p = 1
    opt_gamma = params[1]["gamma"]
    opt_beta = params[1]["beta"]

    # Pull backend information for transpilation once.
    if (not QCTRL) and hardware:
        if isinstance(session, Backend):
            backend = session
        else:
            backend = QiskitRuntimeService().backend(session.backend())
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

    for i, p in enumerate(p_values):
        results[p] = {}
        init_params = np.concatenate((opt_beta, opt_gamma))
        qaoa_circuits[p] = qaoa_circuit_fn(hamiltonian, hamiltonian.num_qubits, p)

        if (not QCTRL) and hardware:
            # Transpile circuits
            circ = pm.run(qaoa_circuits[p])
            transpiled_qaoa_circuits[p] = circ
        else:
            transpiled_qaoa_circuits[p] = qaoa_circuits[p]

        if training:
            if use_estimator:
                if hardware:
                    estimator = Estimator(mode=session)
                    estimator.options.default_shots = 1024

                    # Set simple error suppression/mitigation options
                    estimator.options.dynamical_decoupling.enable = True
                    estimator.options.dynamical_decoupling.sequence_type = "XY4"
                    estimator.options.twirling.enable_gates = True
                    estimator.options.twirling.num_randomizations = "auto"

                    # Transpile the Hamiltonian observable.
                    layout = transpiled_qaoa_circuits[p].layout
                    transpiled_hamiltonian = hamiltonian.apply_layout(layout)
                else:
                    estimator = AerEstimator()
                    transpiled_hamiltonian = hamiltonian
            else:
                if hardware:
                    sampler = Sampler(mode=session)

                    sampler.options.dynamical_decoupling.enable = True
                    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
                    sampler.options.twirling.enable_measure = True
                else:
                    sampler = AerSampler()

                # For the sampler, we don't transpile the Hamiltonain. This is because the sampled
                # bitstrings will automatically have the correct form, because we transpiled the
                # circuit WITH the measurement gates.

                transpiled_hamiltonian = hamiltonian

            Result = minimize(
                cost_func_estimator if use_estimator else cost_func_sampler,
                init_params,
                args=(
                    transpiled_qaoa_circuits[p],
                    transpiled_hamiltonian,
                    estimator if use_estimator else sampler,
                    QCTRL,
                ),
                method="cobyla",
                options={"maxiter": 10 if hardware else 50000},
            )

            # Store optimized parameters
            results[p] = {
                "beta": Result.x[:p],
                "gamma": Result.x[p:],
                "func": Result.fun,
                "nfev": Result.nfev,
            }

            if p < p_final:
                opt_gamma, opt_beta = interpolate_initialize(
                    results[p]["gamma"], results[p]["beta"], p_values[i + 1]
                )

        else:
            results[p]["gamma"] = params[p]["gamma"]
            results[p]["beta"] = params[p]["beta"]

    return results, transpiled_qaoa_circuits


def cost_func_sampler(
    params: np.ndarray,
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    sampler: Sampler,
    QCTRL: bool = False,
) -> float:
    """
    Function to be passed to scipy minimize. Use the qiskit Sampler to evaluate the
    expectation value of the cost Hamiltonian. We can adapt this later to use quantum hardware.

    Args:
        params (np.ndarray): The parameters to bind to the circuit.
        ansatz (QuantumCircuit): The QAOA circuit.
        hamiltonian (SparsePauliOp): The Hamiltonian encoding the cost function.
        sampler (Sampler): An Sampler to sample from the circuit.
    Returns:
        cost (float): The expectation value of the cost Hamiltonian.
    """

    if not QCTRL:
        job = sampler.run([(ansatz, params)], shots=1024)
        sampler_result = job.result()
    else:
        catalog = QiskitFunctionsCatalog()
        perf_mgmt = catalog.load("q-ctrl/performance-management")

        sampler_pubs = [(ansatz, params)]
        sampler_result = perf_mgmt.run(
            primitive="sampler",
            pubs=sampler_pubs,
            instance="partner-eon/all-backends/default",
            backend_name=sampler.backend().name,  # E.g. "ibm_kyiv", or omit to default to the least busy device
        ).result()

    sampled = sampler_result[0].data.c.get_counts()
    total_counts = sum(sampled.values())

    evaluated = {
        state: (count / total_counts, evaluate_hamiltonian_value(hamiltonian, state))
        for state, count in sampled.items()
    }

    cost = sum(probability * value for probability, value in evaluated.values())

    return cost


def cost_func_estimator(
    params: np.ndarray,
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    estimator: Estimator,
    QCTRL: bool = False,
) -> float:
    """
    Function to be passed to scipy minimize. Use the qiskit estimator to evaluate the
    expectation value of the cost Hamiltonian. We can adapt this later to use quantum hardware.

    Args:
        params (np.ndarray): The parameters to bind to the circuit.
        ansatz (QuantumCircuit): The QAOA circuit.
        hamiltonian (SparsePauliOp): The Hamiltonian encoding the cost function.
        sampler (Sampler): An Sampler to sample from the circuit.
    Returns:
        cost (float): The expectation value of the cost Hamiltonian.
    """
    if not QCTRL:
        job = estimator.run([(ansatz, hamiltonian, params)])
        estimator_result = job.result()
    else:
        catalog = QiskitFunctionsCatalog()
        perf_mgmt = catalog.load("q-ctrl/performance-management")

        estimator_pubs = [(ansatz, hamiltonian, params)]
        estimator_result = perf_mgmt.run(
            primitive="estimator",
            pubs=estimator_pubs,
            instance="partner-eon/all-backends/default",
            backend_name=estimator.backend().name,  # E.g. "ibm_kyiv", or omit to default to the least busy device
        ).result()

    return float(estimator_result[0].data.evs)


def evaluate_hamiltonian_value(hamiltonian: SparsePauliOp, bitstring: str) -> complex:
    """
    Given a Hamiltonian (SparsePauliOp) and a bitstring, evaluate the value of the Hamiltonian.

    Parameters:
        hamiltonian (SparsePauliOp): The Hamiltonian represented as a SparsePauliOp.
        bitstring (str): A bitstring (e.g., '0101') representing the quantum state.

    Returns:
        complex: The value of the Hamiltonian for the given bitstring.
    """
    # Initialize Hamiltonian value
    hamiltonian_value = 0

    # Convert bitstring to integer for easier manipulation
    state = int(bitstring, 2)

    # Iterate over each Pauli term and its coefficient in the Hamiltonian
    for pauli_term, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        # Convert the Pauli term to a string (e.g., 'ZIIII', 'IZIII', etc.)
        pauli_str = pauli_term.to_label()

        # Initialize term value to 1 (identity operator contribution)
        term_value = 1

        # Iterate over the Pauli string and calculate the contribution of each Pauli operator
        for n, pauli in enumerate(pauli_str):
            qubit_idx = len(pauli_str) - n - 1  # Little-endian notation
            if pauli == "Z":
                # Apply Z operator: flip phase depending on the corresponding bit
                # Z = -1 when the bit is 1, +1 when the bit is 0
                term_value *= (-1) ** ((state >> qubit_idx) & 1)
            elif pauli == "I":
                # Identity operator: does nothing to the bit
                continue

        # Multiply the term value by the coefficient and add to the Hamiltonian total
        hamiltonian_value += coeff * term_value

    return hamiltonian_value


def is_valid_matching(edges):
    """Check if a list of edges is a valid matching."""
    nodes_in_matching = set()  # Set to track nodes already used in the matching
    for u, v in edges:
        if u in nodes_in_matching or v in nodes_in_matching:
            return False  # If any node is reused, it's not a valid matching
        nodes_in_matching.add(u)
        nodes_in_matching.add(v)
    return True


def compute_qaoa_matching(
    hamiltonian: SparsePauliOp,
    edge_to_qubit: dict,
    p_final: int,
    steps: int,
    iterative_qaoa_optimization_fn: Callable,
    qaoa_circuit_fn: Callable,
    num_valid_matchings: int,
    nx_weight: float,
    weights: Dict,
    use_estimator: bool = True,
    QCTRL: bool = False,
    training: bool = True,
    session: Session = None,
    params: Dict = None,
) -> List[Tuple]:
    """
    Function to find a maximally-weighted matching in a graph.

    Args:
        hamiltonian (SparsePauliOp): The cost Hamiltonian
        edge_to_qubit: (dict): The mapping between edges in the graph and qubits.
        p_final (int): The maximum number of layers in the QAOA circuit.
        steps (int): Starting at 1, how many steps to reach p_final.
        iterative_qaoa_optimization_fn (Callable): Function to teratively optimize QAOA parameters
            for increasing circuit depths.
        qaoa_circuit_fn (Callable): Function to generate the QAOA circuit.
        num_valid_matchings (int): Return the `num_valid_matchings` highest-weight matchings.
        nx_weight (float): The weight of the NetworkX solution
        weights (dict): The edge weights of the graph.
        use_estimator (bool): Use estimator(sampler) for training if True(False).
        QCTRL (bool): Use QCTRL for transpilation if True.
        training (bool): Whether or not to train the QAOA parameters.
        session (Session): If a session is provided, circuits will be executed on hardware.
            NOTE: You can run the jobs in "job" mode by passing a Backend as the `session` argument.
        params (Dict): Dictionary containing beta and gammma values, of the form:
            params = {
                p0: {"beta": ..., "gamma": ...},
                ...,
                p_final: {"beta": ..., "gamma": ...},
            }
            The beta and gamma entries must be np.ndarrays of length equal to the given value of p.
            If training==False, these parameters will be used as specified. If training==True, only
            the p=1 parameters will be used as initialisation.
    Returns:
        matching (List[Tuple]): The minimum-cost matching obtained from QAOA.
    """
    hardware = True if session is not None else False

    results, qaoa_circuits = iterative_qaoa_optimization_fn(
        qaoa_circuit_fn=qaoa_circuit_fn,
        hamiltonian=hamiltonian,
        p_final=p_final,
        steps=steps,
        use_estimator=use_estimator,
        QCTRL=QCTRL,
        training=training,
        session=session,
        params=params,
    )

    p_values = np.linspace(1, p_final, steps, dtype=int)
    matching = {}
    Valid_indicies = {}
    Success_probability = {}
    Ar = {}
    Valid_counts = {}
    weight_matchings = {}
    cost = {}

    sampler_options = SamplerOptions()
    if hardware:
        sampler_options.dynamical_decoupling.enable = True
        sampler_options.dynamical_decoupling.sequence_type = "XpXm"
        sampler_options.twirling.enable_measure = True
        sampler = Sampler(mode=session, options=sampler_options)
    else:
        sampler = AerSampler()

    for p in p_values:
        optimised_qc = qaoa_circuits[p].assign_parameters(
            np.concatenate((results[p]["beta"], results[p]["gamma"]))
        )

        samp_dist = (
            sampler.run([optimised_qc], shots=int(1e4)).result()[0].data.c.get_counts()
        )

        energies = {
            state: (evaluate_hamiltonian_value(hamiltonian, state), count)
            for state, count in samp_dist.items()
        }
        total_counts = sum(count for _, count in energies.values())
        # print("Total count sum:", total_counts)
        min_item = min(energies.items(), key=lambda x: x[1][0])
        # print("Min Energy:", min_item)

        # best_bitstring = min(energies, key=energies.get)

        # sorted_states = sorted(energies, key=energies.get)
        sorted_states_with_counts = sorted(
            energies.items(), key=lambda item: item[1][0]
        )

        valid_matchings_for_p = []
        valid_indices_for_p = []
        weight_matchings_for_p = []
        cost_p = []
        approximation_ratio_numerator = 0  # Sum(weight * count)
        total_valid_counts = 0  # Sum of counts of valid matchings
        sp = 0  # Success probability numerator (count of optimal matchings)
        # print(">>> nx", nx_solution)

        for index, (state, (energy, count)) in enumerate(sorted_states_with_counts):
            matching_candidate = bitstring_to_matching_with_mapping(
                state, edge_to_qubit
            )

            if is_valid_matching(matching_candidate):
                valid_matchings_for_p.append(matching_candidate)
                valid_indices_for_p.append(index)
                weight = calculate_matching_weight(matching_candidate, weights)
                weight_matchings_for_p.append((weight, count))

                cost_p.append((energy, count))

                # Check if this matching has the optimal weight
                if np.isclose(weight, nx_weight):
                    sp += count  # Accumulate count of optimal solutions

                if state == min_item[0]:
                    print(weight, nx_weight)

                # Compute approximation ratio numerator
                approximation_ratio_numerator += weight * count
                total_valid_counts += count  # Track total valid matchings count

        # Compute approximation ratio
        if total_valid_counts > 0:
            approximation_ratio = approximation_ratio_numerator / (
                nx_weight * total_valid_counts
            )
            SP = sp / total_valid_counts  # Compute success probability
        else:
            approximation_ratio = 0
            SP = 0  # Avoid division by zero

        matching[p] = valid_matchings_for_p[:num_valid_matchings]

        weight_matchings[p] = weight_matchings_for_p
        Valid_indicies[p] = valid_indices_for_p
        Ar[p] = approximation_ratio
        Success_probability[p] = SP
        Valid_counts[p] = total_valid_counts
        cost[p] = cost_p
    return (
        matching,
        Valid_indicies,
        Success_probability,
        Ar,
        Valid_counts,
        results,
        weight_matchings,
        cost,
    )
