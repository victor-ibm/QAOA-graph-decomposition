# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from typing import Tuple


def interpolate_initialize(
    opt_gamma: np.ndarray, opt_beta: np.ndarray, p_new: int
) -> Tuple[np.ndarray]:
    """
    Interpolates QAOA parameters to a higher depth with scaling.

    The pre-factor ensures that as the number of parameters increases
    (when the depth of the QAOA circuit increases), the values of the interpolated
    parameters are scaled accordingly. This scaling ensures that the "smoothness" of
    the transition between old and new depths is preserved

    Parameters:
        opt_gamma (np.ndarray): Optimized gamma parameters from depth p.
        opt_beta (np.ndarray): Optimized beta parameters from depth p.
        p_new (int): New depth for interpolation.

    Returns:
        new_gamma (np.ndarray): Interpolated gamma parameters for depth p_new.
        new_beta (np.ndarray): Interpolated beta parameters for depth p_new.
    """
    p_old = len(opt_gamma)

    # Create evenly spaced points for old and new depths
    x_old = np.linspace(0, 1, p_old)
    x_new = np.linspace(0, 1, p_new)

    # Interpolate gamma and beta with scaling
    new_gamma = np.interp(x_new, x_old, opt_gamma)
    new_beta = np.interp(x_new, x_old, opt_beta)

    return new_gamma, new_beta


def fourier_initialize(
    p: int, q: int, u: np.ndarray = None, v: np.ndarray = None
) -> Tuple[np.ndarray]:
    """
    Fourier initialization for QAOA parameters.

    Parameters:
        p (int): Number of QAOA layers.
        q (int): Number of Fourier terms (should be <= p).
        u (np.ndarray): Fourier coefficients for gamma (size q). If None, initialized randomly.
        v (np.ndarray): Fourier coefficients for beta (size q). If None, initialized randomly.

    Returns:
        gamma (np.ndarray): Initialized gamma parameters (size p).
        beta (np.ndarray): Initialized beta parameters (size p).
    """
    # Initialize Fourier coefficients if not provided
    if u is None:
        u = np.random.uniform(-1, 1, q)
    if v is None:
        v = np.random.uniform(-1, 1, q)

    # Precompute the scaling factors
    factor = np.pi / p
    indices = np.arange(1, p + 1) - 0.5

    # Compute gamma using the sine transform
    gamma = np.zeros(p)
    for k in range(1, q + 1):
        gamma += u[k - 1] * np.sin((k - 0.5) * indices * factor)

    # Compute beta using the cosine transform
    beta = np.zeros(p)
    for k in range(1, q + 1):
        beta += v[k - 1] * np.cos((k - 0.5) * indices * factor)

    return gamma, beta
