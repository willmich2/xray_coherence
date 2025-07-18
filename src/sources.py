import torch # type: ignore
from src.simparams import SimParams
torch.pi = torch.acos(torch.zeros(1)).item() * 2

def gaussian_source(sim_params: SimParams, rsrc: float) -> torch.Tensor:
  # Gaussian source
    U_init = torch.zeros((len(sim_params.weights), sim_params.Ny, sim_params.Nx), dtype=torch.complex64, device=sim_params.device)
    R_sq = sim_params.X**2 + sim_params.Y**2
    U_init = torch.exp(-R_sq / (2 * rsrc**2)) * torch.exp(1j * 2*torch.pi * torch.rand(U_init.shape, dtype=torch.float32, device=sim_params.device))

    return U_init

def plane_wave(sim_params: SimParams) -> torch.Tensor:
    U_init = torch.ones((len(sim_params.weights), sim_params.Ny, sim_params.Nx), dtype=torch.complex64, device=sim_params.device)

    return U_init