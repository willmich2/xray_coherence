import torch # type: ignore
from src.simparams import SimParams
torch.pi = torch.acos(torch.zeros(1)).item() * 2

def gaussian_source(params: SimParams, rsrc: float) -> torch.Tensor:
  # Gaussian source
    U_init = torch.zeros((params.Ny, params.Nx), dtype=torch.complex64, device=params.device)
    R_sq = params.X**2 + params.Y**2
    U_init = torch.exp(-R_sq / (2 * rsrc**2)) * torch.exp(1j * 2*torch.pi * torch.rand(U_init.shape, dtype=torch.float32, device=params.device))

    return U_init

def plane_wave(params: SimParams) -> torch.Tensor:
    U_init = torch.ones((params.Ny, params.Nx), dtype=torch.complex64, device=params.device)

    return U_init