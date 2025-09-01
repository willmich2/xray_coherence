import torch # type: ignore
import numpy as np # type: ignore
import scipy.special # type: ignore
from scipy.special import eval_hermite, gammaln # type: ignore
from src.simparams import SimParams
from src.propagation import propagate_z
from src.elements import ArbitraryElement

torch.pi = torch.acos(torch.zeros(1, dtype=torch.float64, device=torch.device("cuda"))).item() * 2

def gaussian_schell_propagate_accumulate_intensity(
    sim_params: SimParams,
    lam_n: torch.Tensor,
    psi_n: torch.Tensor,
    z1: float,
    z2: float, 
    element: ArbitraryElement
) -> torch.Tensor:
    """
    Propagate a Gaussian Schell-model source through a distance z1, apply an arbitrary element, and propagate a distance z2.
    """
    n = lam_n.shape[0]
    i_final = torch.zeros((sim_params.Ny, sim_params.Nx), dtype=torch.float32, device=sim_params.device)

    for wvl in range(sim_params.weights.shape[0]):
        i_wvl = torch.zeros((sim_params.Ny, sim_params.Nx), dtype=torch.float32, device=sim_params.device)
        sim_params_wvl = sim_params.copy()
        sim_params_wvl.lams = sim_params.lams[wvl].unsqueeze(0)
        sim_params_wvl.weights = sim_params.weights[wvl].unsqueeze(0)
        for i in range(n):
            u_init = psi_n[i, :, :].unsqueeze(0)
            assert u_init.shape[0] == 1, "u_init must be a single wavelength"
            u_z1 = propagate_z(u_init, z1, sim_params_wvl)
            u_z1g = element.apply_element(u_z1, sim_params_wvl)
            u_final = propagate_z(u_z1g, z2, sim_params_wvl).squeeze(0)

            i_wvl += torch.abs(u_final * lam_n[i])**2

        i_final += i_wvl * sim_params.weights[wvl]
    
    return i_final

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

    const_factor = ((2 * c) / torch.pi)**0.25
    
    log_n_factor = -0.5 * (n_col * torch.log(torch.tensor(2, dtype=torch.float64, device=narr.device)) + torch.tensor(gammaln(n_col.cpu() + 1), dtype=torch.complex128, device=narr.device))
    n_factor = torch.exp(log_n_factor)

    x_scaled = x * torch.sqrt(torch.tensor(2 * c, dtype=torch.float64, device=narr.device))
    hermite_term = torch.tensor(eval_hermite(n_col.cpu(), x_scaled.cpu()), dtype=torch.complex128, device=narr.device)

    exp_term = torch.exp(-c * x**2)

    phi = const_factor * n_factor * hermite_term * exp_term

    return phi.unsqueeze(1)

def lambda_n(
    rsrc: float, 
    lc: float,
    narr: torch.Tensor, 
) -> torch.Tensor:
    a = 1 / (4*rsrc**2)
    b = 1 / (2*lc**2)
    c = (a**2 + 2*a*b)**(1/2)

    const = torch.tensor((np.pi / (a + b + c))**(1/2), dtype=torch.float32, device=narr.device).unsqueeze(-1)
    exp = torch.tensor(b / (a + b + c), dtype=torch.float32, device=narr.device).pow(narr)
    lam_arr = const * exp
    lam_arr /= lam_arr.sum()
    return lam_arr