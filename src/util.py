import numpy as np # type: ignore
from typing import Tuple

def kramers_law_weights(
        e_min: float,
        e_max: float,
        N: int,
        uniform_energy: bool = True # if True, the energy is sampled uniformly, otherwise the wavelength is sampled uniformly
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
    
    # if uniform_energy is True, the energy is sampled uniformly, otherwise the wavelength is sampled uniformly
    if uniform_energy:
        energies  = np.linspace(e_min, e_max, N)
        lams = h * c / energies
        weights = (lams / lam_min - 1) / lams**2
    else:
        lams = np.linspace(lam_min, lam_max, N)
        energies = h * c / lams
        weights = (energies / e_min - 1) / energies**2
    # ensure weights sum to 1
    weights /= np.sum(weights)
    return lams, weights
