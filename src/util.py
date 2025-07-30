import numpy as np # type: ignore
import pandas as pd # type: ignore
import torch # type: ignore
from typing import Tuple


al_data_energies = np.array([
    1.000e-03, 1.500e-03, 1.560e-03, 1.560999999999999e-3, 2.000e-03, 3.000e-03,
    4.000e-03, 5.000e-03, 6.000e-03, 8.000e-03, 1.000e-02, 1.500e-02,
    2.000e-02, 3.000e-02, 4.000e-02, 5.000e-02, 6.000e-02, 8.000e-02,
    1.000e-01, 1.500e-01, 2.000e-01
]) * 1e6 # convert to eV

al_mass_att_coeffs = np.array([
    1.185e+03, 4.023e+02, 3.621E+03, 3.957e+03, 2.263e+03, 7.881e+02,
    3.605e+02, 1.934e+02, 1.153e+02, 5.032e+01, 2.621e+01, 7.955e+00,
    3.442e+00, 1.128e+00, 5.684e-01, 3.681e-01, 2.778e-01, 2.018e-01,
    1.704e-01, 1.378e-01, 1.223e-01
]) * 270 # convert to m^-1


def kramers_law_weights(
        e_min: float,
        e_max: float,
        N: int,
        filter_al: bool = True,
        filter_thickness: float = 1e-3,
        uniform_energy: bool = True, # if True, the energy is sampled uniformly, otherwise the wavelength is sampled uniformly
        device: torch.device = torch.device("cpu")
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
        if filter_al:
                interp_coeffs = np.interp(energies, al_data_energies, al_mass_att_coeffs)
                weights = np.exp(-filter_thickness*interp_coeffs) * weights
    else:
        lams = np.linspace(lam_min, lam_max, N)
        weights = (lams / lam_min - 1) / lams**2
        if filter_al:
                al_data_lams = np.flip(h * c / al_data_energies)
                interp_coeffs = np.interp(lams, al_data_lams, al_mass_att_coeffs)
                weights = np.exp(-filter_thickness*interp_coeffs) * weights
    # ensure weights sum to 1
    weights /= np.sum(weights)
#     # if a weight is less than weight_cutoff, remove it from the list
#     lams = lams[weights > weight_cutoff]
#     weights = weights[weights > weight_cutoff]

    return torch.tensor(lams, dtype=torch.float32, device=device), torch.tensor(weights, dtype=torch.float32, device=device)


def quasi_monochromatic_spectrum(
        central_energy_ev: float, 
        N: int, 
        bandwidth: float,
        device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates wavelengths and weights for a quasi-monochromatic spectrum.

    This function is useful for simulations where a narrow, uniform energy band
    needs to be modeled as a discrete set of wavelengths.

    Args:
        central_energy_ev (float): The central energy of the spectrum in electron volts (eV).
        num_wavelengths (int): The number of discrete wavelengths to return.
                               Must be a positive integer.
        bandwidth (float): The total relative bandwidth (e.g., 0.01 for 1%).
                           This is centered around the central energy.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - wavelengths_m: An array of wavelengths in meters.
            - weights: An array of corresponding spectral weights. Assumes a
                       uniform (top-hat) distribution, so all weights are equal
                       and sum to 1.0.
    """
    h = 4.135667696e-15 # eV s
    c = 299792458 # m/s

    energy_spread_ev = central_energy_ev * bandwidth
    min_energy_ev = central_energy_ev - energy_spread_ev / 2.0
    max_energy_ev = central_energy_ev + energy_spread_ev / 2.0

    # Create a linearly spaced array of energies across the band
    # If only one wavelength is requested, use the central energy
    if N == 1:
        energies_ev = np.array([central_energy_ev])
    else:
        energies_ev = np.linspace(min_energy_ev, max_energy_ev, N)

    wavelengths_m = (h * c) / energies_ev

    weights = np.ones(N) / N # uniform distribution

    return torch.tensor(wavelengths_m, dtype=torch.float32, device=device), torch.tensor(weights, dtype=torch.float32, device=device)



def create_material_map(
        material_name: str, 
) -> list[np.ndarray]:
   df_k = pd.read_csv(f"/home/gridsan/wmichaels/xray_coherence/data/{material_name}_k.csv")
   df_n = pd.read_csv(f"/home/gridsan/wmichaels/xray_coherence/data/{material_name}_n.csv")
   wavelengths = df_k["wl"].to_numpy() / 1e6 # convert to m
   k = df_k["k"].to_numpy()
   n = df_n["n"].to_numpy()
   return [wavelengths, n + 1j*k]


def refractive_index_at_wvl(
        wvl: torch.Tensor, 
        material_map: list[np.ndarray], 
) -> torch.Tensor:
        wavelengths = material_map[0]
        refractive_indices = material_map[1]
        wvl_np = wvl.cpu().numpy()
        return torch.tensor(np.interp(wvl_np, wavelengths, refractive_indices), dtype=torch.complex64, device=wvl.device)
