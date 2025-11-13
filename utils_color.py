import numpy as np

def deltaE76(Lab1, Lab2):
    """
    Compute CIE76 ΔE between Lab1 and Lab2.

    Lab1, Lab2:
      - can be 1D (3,) or 2D (N,3)
    Returns:
      - 1D array of distances (length N) or scalar if inputs are 1D.
    """
    L1 = np.asarray(Lab1, dtype=float)
    L2 = np.asarray(Lab2, dtype=float)

    # Ensure at least 2D: (N,3)
    if L1.ndim == 1:
        L1 = L1.reshape(1, -1)
    if L2.ndim == 1:
        L2 = L2.reshape(1, -1)

    diff = L1 - L2
    dE = np.sqrt(np.sum(diff**2, axis=-1))

    # If it was single pair, return scalar
    if dE.size == 1:
        return float(dE[0])
    return dE
