import numpy as np # type: ignore
from typing import Tuple

def kramers_law_weights(
        e_min: float,
        e_max: float,
        N: int
) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Calculate the spectral weights according to Kramer's law. The input energies are in eV and 
    the output should be wavelengths and weights in inverse meters.
    """
    # convert to angular frequencies    
    h = 4.135667696e-15 # eV s
    c = 299792458 # m/s
    lam_min = h * c / e_max
    lam_max = h * c / e_min

    lams = np.linspace(lam_min, lam_max, N)
    weights = (lams / lam_min - 1) / lams**2
    # ensure weights sum to 1
    weights /= np.sum(weights)
    return lams, weights
