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

w_data_energies = np.array([
    1.00000e-03, 1.50000e-03, 1.80920e-03, 1.84000e-03, 1.87160e-03,
       1.87160e-03, 1.91960e-03, 2.00000e-03, 2.28100e-03, 2.28100e-03,
       2.42350e-03, 2.57490e-03, 2.57490e-03, 2.69447e-03, 2.81960e-03,
       2.81960e-03, 3.00000e-03, 4.00000e-03, 5.00000e-03, 6.00000e-03,
       8.00000e-03, 1.00000e-02, 1.02068e-02, 1.02068e-02, 1.08548e-02,
       1.15440e-02, 1.15440e-02, 1.18186e-02, 1.20990e-02, 1.20990e-02,
       1.50000e-02, 2.00000e-02, 3.00000e-02, 4.00000e-02, 5.00000e-02,
       6.00000e-02, 6.95250e-02, 6.95250e-02, 8.00000e-02, 1.00000e-01,
       1.50000e-01, 2.00000e-01
]) * 1e6 # convert to eV

w_mass_att_coeffs = np.array([
    3.683e+03, 1.643e+03, 1.108e+03, 1.927e+03, 1.991e+03, 2.901e+03,
       3.149e+03, 3.922e+03, 2.828e+03, 3.279e+03, 2.833e+03, 2.485e+03,
       3.599e+03, 2.339e+03, 2.104e+03, 2.193e+03, 1.902e+03, 9.564e+02,
       5.534e+02, 3.514e+02, 1.786e+02, 9.691e+01, 9.201e+01, 2.134e+02,
       1.983e+02, 1.689e+02, 2.311e+02, 2.268e+02, 2.065e+02, 2.332e+02,
       1.389e+02, 6.512e+01, 2.273e+01, 1.067e+01, 5.940e+00, 3.713e+00,
       2.552e+00, 1.175e+01, 7.810e+00, 4.438e+00, 1.481e+00, 7.844e-01
]) * 1930 # convert to m^-1

def kramers_law_weights(
        e_min: float,
        e_max: float,
        N: int,
        filter_weights: bool = True,
        filter_thickness: float = 1e-3,
        filter_material: str = "al",
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

    # Build a dense spectrum first, then downsample centered around the central wavelength
    dense_N = max(2048, 10 * max(1, N))

    if uniform_energy:
        energies_dense = np.linspace(e_min, e_max, dense_N)
        lams_dense = h * c / energies_dense
        weights_dense = e_max / energies_dense - 1
        if filter_weights:
            if filter_material == "al":
                interp_coeffs = np.interp(energies_dense, al_data_energies, al_mass_att_coeffs)
                weights_dense = np.exp(-filter_thickness * interp_coeffs) * weights_dense
            elif filter_material == "w":
                interp_coeffs = np.interp(energies_dense, w_data_energies, w_mass_att_coeffs)
                weights_dense = np.exp(-filter_thickness * interp_coeffs) * weights_dense
        lams_dense = np.flip(lams_dense) if lams_dense[0] > lams_dense[-1] else lams_dense
        weights_dense = weights_dense[::-1] if lams_dense[0] > lams_dense[-1] else weights_dense
    else:
        lams_dense = np.linspace(lam_min, lam_max, dense_N)
        weights_dense = (lams_dense / lam_min - 1) / lams_dense**2
        if filter_weights:
            if filter_material == "al":
                al_data_lams = np.flip(h * c / al_data_energies)
                interp_coeffs = np.interp(lams_dense, al_data_lams, al_mass_att_coeffs)
                weights_dense = np.exp(-filter_thickness * interp_coeffs) * weights_dense
            elif filter_material == "w":
                w_data_lams = np.flip(h * c / w_data_energies)
                interp_coeffs = np.interp(lams_dense, w_data_lams, w_mass_att_coeffs)
                weights_dense = np.exp(-filter_thickness * interp_coeffs) * weights_dense

    # Normalize dense weights
    weights_dense = weights_dense / np.sum(weights_dense)

    # lam_center is wavelength of maximum weight
    lam_center = lams_dense[np.argmax(weights_dense)]
    center_idx = int(np.argmin(np.abs(lams_dense - lam_center)))

    if N == 1:
        lams = np.array([lams_dense[center_idx]])
        weights = np.array([1.0])
    else:
        # Choose a regular step and start so samples are centered around lam_center
        step = max(1, int(np.floor(dense_N / N)))
        if N % 2 == 1:
            start = center_idx - (N // 2) * step
        else:
            start = int(np.round(center_idx - (N - 1) * 0.5 * step))
        start = max(0, min(start, dense_N - 1 - (N - 1) * step))
        idxs = start + step * np.arange(N)
        idxs = np.clip(idxs, 0, dense_N - 1).astype(int)
        lams = lams_dense[idxs]
        weights = weights_dense[idxs]
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


def spherize_1d_array(radial_profile: np.ndarray) -> np.ndarray:
    """
    Expands a 1D radial profile into a 2D array with circular symmetry.

    This function takes a 1D array, representing function values along a radius,
    and generates a 2D square array where the value of each pixel is determined
    by its radial distance from the center. This effectively "rotates" the 1D
    profile around the center to fill a 2D space.

    Args:
        radial_profile: A 1D NumPy array of size N representing the function's
                        values along a radius.

    Returns:
        A 2D NumPy array of shape (2*N-1, 2*N-1) representing the
        circularly symmetric function.
    """
    # Get the radius N from the length of the input array.

    if radial_profile.ndim != 1:
        radial_profile = radial_profile.reshape(-1)
    n = radial_profile.shape[0]
    
    # The output 2D array will have a side length of 2*N - 1 to ensure a single center pixel.
    diameter = 2 * n - 1
    
    # Define the center of the 2D array.
    center_x, center_y = n - 1, n - 1

    # Create coordinate grids.
    # np.arange(diameter) creates an array [0, 1, ..., diameter-1].
    # Subtracting the center coordinate shifts the grid so the center is at (0,0).
    x = np.arange(diameter) - center_x
    y = np.arange(diameter) - center_y
    xx, yy = np.meshgrid(x, y)

    # Calculate the Euclidean distance (radius) from the center for every point in the grid.
    radius_grid = np.sqrt(xx**2 + yy**2)

    # Round the distances to the nearest integer to use them as indices
    # for the input radial_profile array.
    index_grid = np.round(radius_grid).astype(int)

    # Create an output array, initialized with a fill value (e.g., 0).
    # The data type is matched to the input array's type.
    # Using float allows for potential future use with NaN or interpolation.
    output_2d = np.zeros((diameter, diameter), dtype=radial_profile.dtype)

    # Create a mask for all pixels that are within the original radius N.
    valid_mask = index_grid < n

    # Use the index_grid to look up the corresponding values from the radial_profile.
    # This is an advanced indexing operation. For every `True` position in `valid_mask`,
    # we take the corresponding integer from `index_grid` and use it as an index
    # into `radial_profile`. The resulting value is placed in the `output_2d` array.
    output_2d[valid_mask] = radial_profile[index_grid[valid_mask]]

    return output_2d