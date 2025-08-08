import torch # type: ignore
import numpy as np # type: ignore
from src.simparams import SimParams
from src.sparse import matrix_free_eigsh
import scipy.special # type: ignore

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def gaussian_source(sim_params: SimParams, rsrc: float) -> torch.Tensor:
  # Gaussian source
    U_init = torch.zeros((len(sim_params.weights), sim_params.Ny, sim_params.Nx), dtype=sim_params.dtype, device=sim_params.device)
    R_sq = sim_params.X**2 + sim_params.Y**2
    U_init = torch.exp(-R_sq / (2 * rsrc**2)) * torch.exp(1j * 2*torch.pi * torch.rand(U_init.shape, dtype=sim_params.dtype, device=sim_params.device))

    return U_init

def plane_wave(sim_params: SimParams) -> torch.Tensor:
    U_init = torch.ones((len(sim_params.weights), sim_params.Ny, sim_params.Nx), dtype=sim_params.dtype, device=sim_params.device)

    return U_init


def circ_mutual_intensity(
    sim_params: SimParams,
    lam: float, 
    r: float, 
    z: float
):
    X1, X2 = torch.meshgrid((sim_params.x, sim_params.x), indexing='ij')
    DX = (X2 - X1).abs()

    k = 2*np.pi / lam
    psi = np.pi / (lam*z)*(X2.abs().pow(2) - X1.abs().pow(2))

    diagonal_mask = torch.eye(DX.shape[0], dtype=bool, device=DX.device)
    DX[diagonal_mask] = 1.0
    
    J = scipy.special.jv(1, DX.cpu().numpy()*k*r/z) / DX.cpu().numpy()*k*r/z
    J = torch.tensor(J, device=sim_params.device)
    J = torch.exp(-1j*psi)*J
    J /= J.abs().max()
    J.fill_diagonal_(1.)
    return J


def incoherent_source(sim_params: SimParams, rsrc: float, z: float, N: int, sparse_tol: float) -> torch.Tensor:
    modes = torch.zeros((sim_params.weights.shape[0], N, sim_params.Ny, sim_params.Nx), dtype=sim_params.dtype, device=sim_params.device)
    evals = torch.zeros((sim_params.weights.shape[0], N), dtype=sim_params.dtype, device=sim_params.device)

    for i, lam in enumerate(sim_params.lams):
      J = circ_mutual_intensity(sim_params, lam, rsrc, z)

      J[J.abs()< sparse_tol] = 0.0
      
      J_sparse = J.to_sparse()
      
      evals, evecs = matrix_free_eigsh(J_sparse, N)
      
      evals = evals / evals.max()
      modes[i] = evecs
      evals[i] = evals

    modes = modes.transpose(0, 1)
    evals = evals.transpose(0, 1)
    return modes, evals