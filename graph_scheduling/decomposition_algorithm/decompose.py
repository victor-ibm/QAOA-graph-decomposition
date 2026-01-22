# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import time
import numpy as np
from graph_scheduling.decomposition_algorithm.cplex_utils import cplex_minimise_cost
from graph_scheduling.decomposition_algorithm.utils import (
    get_current_demand,
    nx_max_weight_matching,
    qaoa_high_weight_matchings,
    sa_high_weight_matchings,
    random_high_weight_matchings,
    find_unique_matchings_and_weights,
)
from qiskit_ibm_runtime import Session


class Decomposer:
    def __init__(
        self,
        demand_matrix: np.array,
        tol: float,
        method: str = "FCFW",
        renormalise_weights: bool = True,
        e_fcfw_config: dict = {
            "matchings_per_iteration": 5,
            "sampling_method": "QAOA",
            "qaoa_config": {
                "num_qaoa_layers": 1,
                "train_params": True,
                "simulation_method": "simulator",
            },
        },
        session: Session = None,
    ):
        """
        Class to decompose a demand matrix into a combination of matchings.

        There are four main methods for decomposition:

        1.  'BKF': Birkhoff-like algorithm.
            At each stage, the max-weight max-cardinality matching in the current demand matrix
            (original - sum(coeffs * matchings)) is added to the decomposition, and the coefficient
            is chosen to be the minimum value in the current demand matrix, at the locations of the
            edges in the matching. For example:

            matrix:
            [[ 0 , 0.2,  0 ],
            [0.2,  0 , 0.8],
            [ 0 , 0.8,  0 ]]

            matching:
            [[0, 0, 0],
            [0, 0, 1],
            [0, 1, 0]]

            coefficient:
            0.8
        2.  'FW': Frank-Wolfe algorithm.
            At each stage of the decomposition, the maximally-weighted matching is obtained. Then
            the weights are re-evaluated by taking a step in the direction of the maximally-weighted
            matching, according to a step-size strategy of 1/iteration.
        3.  'FCFW': Fully-Corrective Frank-Wolfe algorithm.
            At each stage of the decomposition, the max-weight max-cardinality matching in current
            demand matrix (original - sum(coeffs * matchings)) is added to the decomposition. Then
            all the weights are optimised using CPLEX.
        4.  'E-FCFW': Extended-FCFW algorithm.
            Similar to FCFW, but multiple high-weight matchings are sampled and added to the
            decomposition in each iteration. See the paper for the full algorithm description.

            Sampling methods include: QAOA, random, and simulated-annealing.

        Args:
            demand_matrix (np.ndarray): The demand matrix to be decomposed.
            tol (float): The decomposition will finish when the cost falls below tol.
            method (str): The method for choosing the matching at each stage of the decomposition.
                valid methods are: "BKF", "FW", "FCFW", "E-FCFW"
            renormalise_weights (bool): Whether or not to normalise the weights, so that they sum to
                1 at the end of the decomposition.
            e_fcfw_config (dict): Configuration options for the extended-FCFW method.
            session (Session): The session to use when running on quantum hardware. Only relevant
                when using the E-FCFW method with QAOA sampling on hardware.
        """

        # Configuration options
        self.demand_matrix = demand_matrix
        self.tol = tol
        if method in ["BKF", "FW", "FCFW", "E-FCFW"]:
            self.method = method
        else:
            raise ValueError(
                f"Invalid method: {method}. Must be one of 'BKF', 'FW', 'FCFW', or 'E-FCFW'."
            )
        self.renormalise_weights = renormalise_weights
        if self.method == "E-FCFW":
            # Check that e_fcfw_config is valid.
            assert "matchings_per_iteration" in e_fcfw_config.keys()
            assert "sampling_method" in e_fcfw_config.keys()
            if e_fcfw_config["sampling_method"] not in ["QAOA", "uniform_random", "SA"]:
                raise ValueError(
                    f"Invalid sampling_method: {e_fcfw_config['sampling_method']}. Must be one of 'QAOA', 'uniform_random', or 'SA'."
                )
            self.matchings_per_iteration = e_fcfw_config["matchings_per_iteration"]
            self.sampling_method = e_fcfw_config["sampling_method"]
            # Check that qaoa_config is valid.
            if self.sampling_method == "QAOA":
                assert "qaoa_config" in e_fcfw_config.keys()
                self.qaoa_config = e_fcfw_config["qaoa_config"]
                assert "num_qaoa_layers" in self.qaoa_config.keys()
                assert "train_params" in self.qaoa_config.keys()
                assert "simulation_method" in self.qaoa_config.keys()
                if self.qaoa_config["simulation_method"] not in ["simulator", "hardware"]:
                    raise NotImplementedError()
                if self.qaoa_config["simulation_method"] == "hardware" and (session is None):
                    raise Exception(
                        "You must provide a session when running QAOA on quantum hardware"
                    )
                self.session = session

                if not self.qaoa_config["train_params"]:
                    assert "params" in self.qaoa_config.keys()
                else:
                    self.qaoa_config["params"] = self.qaoa_config.get("params")

        # Initalise decomposition params
        self.coeffs = []
        self.matchings = []
        self.cost = np.sum(np.square(demand_matrix)) / (demand_matrix.shape[0] ** 2)
        self.cost_history = [self.cost]
        self.num_matchings_history = [0]

    def decompose(self):
        """
        Decompose the demand matrix using the specified method.

        Returns:
            results (Dict): A dictionary containing the relevant results.
        """

        start_time = time.time()
        self.iteration = 0
        if self.method == "E-FCFW":
            self.coeffs_fw = []
            self.matchings_fw = []

        # Main algorithm loop.
        while self.cost > self.tol:
            self.iteration += 1
            # Find matching(s).
            self.find_matchings()

            # Recompute the weights.
            self.compute_weights()

            # Update the cost.
            self.cost = self.evaluate_cost()
            self.cost_history.append(self.cost)

            # Group any duplicate matchings.
            self.group_duplicate_matchings()

            # Record the number of matchings with non-zero weight.
            self.num_matchings_history.append(
                len([coeff for coeff in self.coeffs if coeff != 0])
            )

            # Break if the cost stops improving.
            if self.cost_history[-1] == 0.0 or (
                abs(
                    (self.cost_history[-1] - self.cost_history[-2])
                    / self.cost_history[-1]
                )
                < 0.01
            ):
                break

        if self.renormalise_weights:
            print(f"Cost before normalising coefficients: {self.cost_history[-1]}")
            # At the end, normalise coefficients to sum to 1 and evaluate final cost.
            total = np.sum(self.coeffs)
            self.coeffs = [coeff / total for coeff in self.coeffs]

            self.cost = self.evaluate_cost()
            self.cost_history.append(self.cost)
            self.num_matchings_history.append(
                len([coeff for coeff in self.coeffs if coeff != 0])
            )
        print(f"Final cost:  {self.cost_history[-1]}")

        # Remove any 0 coefficients and their corresponding matchings
        for i in list(range(len(self.coeffs)))[::-1]:
            if self.coeffs[i] == 0:
                del self.coeffs[i]
                del self.matchings[i]

        end_time = time.time()

        # Collect results
        results = {
            "weights": self.coeffs,
            "matchings": self.matchings,
            "cost_history": self.cost_history,
            "num_matchings_history": self.num_matchings_history,
            "time_taken": end_time - start_time,
        }

        if (
            self.method == "E-FCFW"
            and self.sampling_method == "QAOA"
            and self.qaoa_config["simulation_method"] == "hardware"
        ):
            self.session.close()
            details = self.session.details()
            # NOTE: This is the total time the QPU is reserved for us, and will include any
            # classical compute time between QPU jobs.
            results["qpu_time"] = details["usage_time"]
            results["session_id"] = details["id"]

        return results

    def find_matchings(self):
        """
        Finds the relevant matching(s) for the specific decomposition method.
        """
        if self.method in ["BKF", "FW", "FCFW"]:
            # Find the maximally weighted matching in (demand - coeffs.matchings)
            current_demand = get_current_demand(
                self.demand_matrix, self.coeffs, self.matchings
            )
            M = nx_max_weight_matching(current_demand)

            self.matchings.append(M)
        elif self.method == "E-FCFW":
            # Find the maximally weighted matching in (demand - coeffs_fw.matchings_fw)
            current_demand_fw = get_current_demand(
                self.demand_matrix, self.coeffs_fw, self.matchings_fw
            )
            m = nx_max_weight_matching(current_demand_fw)

            # Sample high-weight matchings in (demand - coeffs.matchings)
            current_demand = get_current_demand(
                self.demand_matrix, self.coeffs, self.matchings
            )
            if self.sampling_method == "QAOA":
                ms = qaoa_high_weight_matchings(
                    matrix=current_demand,
                    num_matchings=self.matchings_per_iteration,
                    p=self.qaoa_config["num_qaoa_layers"],
                    train_params=self.qaoa_config["train_params"],
                    params=self.qaoa_config["params"],
                    session=self.session,
                )
            elif self.sampling_method == "uniform_random":
                ms = random_high_weight_matchings(
                    current_demand, self.matchings_per_iteration
                )
            elif self.sampling_method == "SA":
                ms = sa_high_weight_matchings(
                    current_demand, self.matchings_per_iteration
                )

            self.matchings_fw.append(m)
            self.matchings += [m] + ms

    def compute_weights(self):
        """
        Recomputes the weights in a method-specific way.
        """
        if self.method == "BKF":
            # Get current demand without newest matching
            current_demand = get_current_demand(
                self.demand_matrix, self.coeffs, self.matchings[:-1]
            )
            # Find the minimum non-zero element at non-zero elements of the newest matching
            select = np.multiply(self.matchings[-1], current_demand)
            coeff = np.min(select[np.nonzero(select)])
            self.coeffs.append(coeff)
        elif self.method == "FW":
            # Step towards the newest matching.
            step = 1 / self.iteration
            self.coeffs = [(1 - step) * coeff for coeff in self.coeffs]
            self.coeffs.append(step)
        elif self.method == "FCFW":
            # Find the optimal weights by minimising ||demand - decomp||^2_F using CPLEX.
            self.coeffs.append(1)
            self.coeffs, cost = cplex_minimise_cost(self.demand_matrix, self.matchings)
        elif self.method == "E-FCFW":
            # Find the optimal weights by minimising ||demand - decomp||^2_F using CPLEX.
            self.coeffs_fw.append(1)
            self.coeffs += [1] * (len(self.matchings) - len(self.coeffs))

            self.coeffs_fw, _ = cplex_minimise_cost(
                self.demand_matrix, self.matchings_fw
            )
            fw_non_zero = len([c for c in self.coeffs_fw if c != 0])
            self.coeffs, _ = cplex_minimise_cost(
                self.demand_matrix, self.matchings, max_length=fw_non_zero
            )

    def evaluate_cost(self):
        """
        Evaluate the cost as ||demand - coeffs.matchings||^2_F / n^2

        Returns:
            cost: The cost.
        """
        current_demand = get_current_demand(
            self.demand_matrix, self.coeffs, self.matchings
        )
        cost = np.sum(np.square(current_demand)) / (self.demand_matrix.shape[0] ** 2)

        return cost

    def group_duplicate_matchings(self):
        """
        If there are any duplicated matchings in self.matchings, this function removes the
        duplicates, and adds the coefficients together. For the E-FCFW method, this also does the
        same for self.matchings_fw.
        """
        self.matchings, self.coeffs = find_unique_matchings_and_weights(
            self.matchings, self.coeffs
        )
        if self.method == "E-FCFW":
            self.matchings_fw, self.coeffs_fw = find_unique_matchings_and_weights(
                self.matchings_fw, self.coeffs_fw
            )
