import numpy as np # type: ignore
import pandas as pd # type: ignore
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
        weights = e_max / energies - 1
    else:
        lams = np.linspace(lam_min, lam_max, N)
        weights = (lams / lam_min - 1) / lams**2
    # ensure weights sum to 1
    weights /= np.sum(weights)
    return lams, weights

def create_material_map(
        material_name: str, 
) -> list[np.ndarray]:
   df_k = pd.read_csv(f"../data/{material_name}_k.csv")
   df_n = pd.read_csv(f"../data/{material_name}_n.csv")
   wavelengths = df_k["wl"].to_numpy()
   k = df_k["k"].to_numpy()
   n = df_n["n"].to_numpy()
   return [wavelengths, n + 1j*k]

def refractive_index_at_wvl(
        wvl: float, 
        material_map: list[np.ndarray], 
) -> complex:
        wavelengths = material_map[0]
        refractive_indices = material_map[1]
        return np.interp(wvl, wavelengths, refractive_indices)

