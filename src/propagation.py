import torch # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np # type: ignore
from src.simparams import SimParams
from src.util import refractive_index_at_wvl

def angular_spectrum_propagation(
    U: torch.Tensor, 
    lam: float, 
    z: float, 
    dx: float, 
    device: torch.device
    ) -> torch.Tensor:
    
    U_padded = pad_double_width(U)

    Ny_padded, Nx_padded = U_padded.shape

    pi = torch.acos(torch.tensor(-1.0, dtype=torch.float32, device=device))
    k0 = 2*pi/lam

    kx = torch.fft.fftfreq(Nx_padded, dx, dtype=torch.float32, device=device) * 2*pi
    ky = torch.fft.fftfreq(Ny_padded, dx, dtype=torch.float32, device=device) * 2*pi
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')

    sqrt_arg = k0**2 - KX**2 - KY**2
    sqrt_arg[sqrt_arg < 0] = 0.0

    transfer_function = torch.exp(1j * z * torch.sqrt(sqrt_arg))


    U_fourier = torch.fft.fft2(U_padded)
    U_z_padded = torch.fft.ifft2(U_fourier * transfer_function)
    U_z = unpad_half_width(U_z_padded)
    return U_z

def propagate_z(
    U: torch.Tensor, 
    z: float, 
    sim_params: SimParams
    ) -> torch.Tensor:
    
    Uz = torch.zeros((len(sim_params.weights), sim_params.Ny, sim_params.Nx), dtype=torch.complex64, device=sim_params.device)

    for i, lam in enumerate(sim_params.lams):
        Uz_lam = angular_spectrum_propagation(U[i, :, :], lam, z, sim_params.dx, sim_params.device)
        Uz[i, :, :] = Uz_lam
    return Uz


def pad_double_width(x: torch.Tensor) -> torch.Tensor:
    """
    Zero‑pad a tensor whose last two dims are (1, W) so the width
    becomes 2 W while the single row stays unchanged.

    Parameters
    ----------
    x : torch.Tensor
        Shape (..., 1, W)

    Returns
    -------
    torch.Tensor
        Shape (..., 1, 2*W) with the input centered horizontally.
    """
    if x.shape[-2] != 1:
        raise ValueError("Row dimension must be 1; only width is padded.")

    W = x.shape[-1]
    pad_left  = W // 2
    pad_right = W - pad_left                        # handles odd W

    # (left, right, top, bottom)
    return F.pad(x, (pad_left, pad_right, 0, 0), mode="constant", value=0)


def unpad_half_width(x: torch.Tensor) -> torch.Tensor:
    """
    Undo `pad_double_width`: crop the central width segment, leaving
    the single row intact.

    Parameters
    ----------
    x : torch.Tensor
        Shape (..., 1, 2*W)

    Returns
    -------
    torch.Tensor
        Shape (..., 1, W)
    """
    if x.shape[-2] != 1 or x.shape[-1] % 2:
        raise ValueError(
            "Input must have shape (..., 1, 2*W) with an even width."
        )

    W = x.shape[-1] // 2
    start = (x.shape[-1] - W) // 2                  # == W//2
    return x[..., :, start : start + W]


def apply_element(
    U: torch.Tensor, 
    element, 
    sim_params: SimParams
    ) -> torch.Tensor:
    U_f = torch.zeros((len(sim_params.weights), sim_params.Ny, sim_params.Nx), dtype=torch.complex64, device=sim_params.device)
    max_lam = sim_params.lams[np.argmax(sim_params.weights)]
    transmission = element.transmission(max_lam, sim_params)

    for i, lam in enumerate(sim_params.lams):
        U_lam = U[i, :, :] * transmission
        U_f[i, :, :] = U_lam
    return U_f


def apply_element_sliced(
    U: torch.Tensor, 
    element, 
    slice_thickness: float,
    sim_params: SimParams
    ) -> torch.Tensor:
    
    U_f = torch.zeros((len(sim_params.weights), sim_params.Ny, sim_params.Nx), dtype=torch.complex64, device=sim_params.device)

    max_lam = sim_params.lams[np.argmax(sim_params.weights)]
    
    for i, lam in enumerate(sim_params.lams):
        t = element.thickness
        n_slices = int(t // slice_thickness)
        for j in range(n_slices):
            t_slice = slice_thickness
            if j == n_slices - 1:
                t_slice = t - j * slice_thickness
            slice_element = element.__copy__()
            slice_element.thickness = t_slice

            transmission = slice_element.transmission(max_lam, sim_params)
            U_lam = U[i, :, :] * transmission
            U_lam = angular_spectrum_propagation(U_lam, lam, t_slice, sim_params.dx, sim_params.device)
            U_f[i, :, :] = U_lam
        
    return U_f