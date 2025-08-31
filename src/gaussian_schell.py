import torch # type: ignore
import numpy as np # type: ignore
import scipy.special # type: ignore
from scipy.special import eval_hermite, gammaln # type: ignore
from src.simparams import SimParams

def gaussian_schell_propagate_accumulate_intensity(
    sim_params: SimParams,
    lc: float,
    rsrc: float,
    z1: float,
    z2: float, 
    nmax: int
) -> torch.Tensor:
    """
    Propagate a Gaussian Schell-model source through a distance z, apply an arbitrary element, and propagate a distance z again.
    """
    n = torch.arange(nmax)
    x = sim_params.x
    lam_n = lambda_n(rsrc, lc, n, tensor_3D=True)
    psi_n = psi_n(rsrc, lc, n, x)

    return True


def psi_n(
    rsrc: float, 
    lc: float, 
    narr: torch.Tensor, 
    x: torch.Tensor
) -> torch.Tensor:
    a = 1 / (4*rsrc**2)
    b = 1 / (2*lc**2)
    c = (a**2 + 2*a*b)**(1/2)

    n_col = narr[:, None]

    const_factor = ((2 * c) / np.pi)**0.25
    
    log_n_factor = -0.5 * (n_col * torch.log(2) + gammaln(n_col + 1))
    n_factor = torch.exp(log_n_factor, dtype=torch.float64)

    x_scaled = x * np.sqrt(2 * c)
    hermite_term = eval_hermite(n_col, x_scaled)

    exp_term = torch.exp(-c * x**2)

    phi = const_factor * n_factor * hermite_term * exp_term

    return phi

def lambda_n(
    rsrc: float, 
    lc: float,
    narr: torch.Tensor, 
) -> torch.Tensor:
    a = 1 / (4*rsrc**2)
    b = 1 / (2*lc**2)
    c = (a**2 + 2*a*b)**(1/2)

    const = (np.pi / (a + b + c)).pow(1/2).unsqueeze(-1)
    exp = (b / (a + b + c)).unsqueeze(-1).pow(narr)
    lam_arr = const * exp
    max_arr = torch.max(lam_arr, dim=2, keepdim=True).values
    
    norm_arr = lam_arr / max_arr
    return norm_arr