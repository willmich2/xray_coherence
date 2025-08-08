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


def circ_mutual_intensity_sparse(
    sim_params: SimParams,
    lam: float, 
    r: float, 
    z: float, 
    sparse_tol: float = 1e-2
):
    """
    Computes a sparse representation of the mutual intensity function without
    allocating the full dense matrix in memory, using vectorized operations for speed.

    This function leverages the fact that the mutual intensity function's
    magnitude decays with distance, meaning the resulting matrix is effectively
    band-diagonal. It calculates the bandwidth of significant values and only
    computes elements within that band using efficient tensor operations.

    Args:
        sim_params: An object containing the 1D coordinate tensor `x` and the device.
        lam: Wavelength of the light.
        r: Radius of the circular aperture.
        z: Propagation distance.
        sparse_tol: The tolerance below which normalized values are considered zero.

    Returns:
        A sparse torch.Tensor (COO format) representing the mutual intensity function.
    """
    # --- 1. Initialization and Constants ---
    device = sim_params.device
    x = sim_params.x
    N = len(x)
    
    if not torch.all(x[1:] >= x[:-1]):
        raise ValueError("Input tensor sim_params.x must be sorted.")

    dx_step = x[1] - x[0]  # Assuming uniform spacing
    k = 2 * torch.pi / lam

    # --- 2. Determine Normalization Factor and Sparsity Threshold ---
    # The maximum absolute value of the off-diagonal elements is determined by the
    # jinc function J1(x)/x, and its peak is near the smallest spatial separation.
    arg_at_min_dx = (dx_step * k * r / z).cpu().numpy()
    
    # The jinc function J1(y)/y approaches 0.5 as y->0.
    if arg_at_min_dx < 1e-6:
        max_abs_val = 0.5
    else:
        max_abs_val = np.abs(scipy.special.jv(1, arg_at_min_dx) / arg_at_min_dx)
    
    # This is the absolute value threshold for the un-normalized jinc values.
    abs_threshold = sparse_tol * max_abs_val

    # --- 3. Calculate Bandwidth of Significant Elements ---
    def func_to_solve(arg, target):
        value = np.abs(scipy.special.jv(1, arg) / arg) if arg > 1e-9 else 0.5
        return value - target

    a = arg_at_min_dx
    b = a + 1.0 
    while func_to_solve(b, abs_threshold) > 0:
        b *= 2.0

    arg_cutoff = scipy.optimize.brentq(func_to_solve, a, b, args=(abs_threshold,))
    
    dx_max = arg_cutoff * z / (k * r)
    bandwidth = int(torch.ceil(dx_max / dx_step))

    # --- 4. Vectorized Construction of Off-Diagonal Elements ---
    # Generate all indices within the band at once
    band_offsets = torch.arange(1, bandwidth + 1, device=device)
    base_indices = torch.arange(N, device=device).unsqueeze(1)
    
    # Create row and column indices for the upper band
    row_indices = base_indices.expand(-1, bandwidth)
    col_indices = row_indices + band_offsets
    
    # Filter out out-of-bounds indices
    valid_mask = col_indices < N
    rows_upper = row_indices[valid_mask]
    cols_upper = col_indices[valid_mask]

    # Calculate distances and arguments for the Bessel function in a vectorized manner
    x1 = x[rows_upper]
    x2 = x[cols_upper]
    dx_vals = x2 - x1
    args = dx_vals * (k * r / z)
    
    # Compute jinc function. This is the main CPU-bound step.
    # We use numpy for this and then transfer back to the device.
    args_cpu = args.cpu().numpy()
    # Use np.divide for safe division, handling the arg=0 case
    jinc_vals_cpu = np.divide(scipy.special.jv(1, args_cpu), args_cpu, where=args_cpu!=0)
    jinc_vals_cpu[args_cpu==0] = 0.5 # Manually set the limit for arg=0
    jinc_vals = torch.from_numpy(jinc_vals_cpu).to(device)
    
    # Calculate phase and combine into complex values
    psi = (torch.pi / (lam.cpu().numpy() * z)) * (x2.pow(2) - x1.pow(2))
    vals_unnormalized = torch.exp(-1j * psi) * jinc_vals
    
    # Normalize and create final values for the sparse tensor
    vals_normalized = vals_unnormalized / max_abs_val

    # --- 5. Assemble Final Sparse Tensor ---
    # Combine diagonal, upper, and lower band elements
    diag_indices = torch.arange(N, device=device)
    
    # Indices for all non-zero elements
    all_indices = torch.cat([
        torch.stack([diag_indices, diag_indices]), # Diagonal
        torch.stack([rows_upper, cols_upper]),     # Upper band
        torch.stack([cols_upper, rows_upper])      # Lower band (Hermitian conjugate)
    ], dim=1)
    
    # Values for all non-zero elements
    all_values = torch.cat([
        torch.ones(N, dtype=torch.complex64, device=device), # Diagonal values are 1
        vals_normalized,                                     # Upper band values
        vals_normalized.conj()                               # Lower band values
    ])
    
    # Create and coalesce the sparse tensor
    J_sparse = torch.sparse_coo_tensor(all_indices, all_values, (N, N))
    return J_sparse.coalesce()

def incoherent_source(sim_params: SimParams, rsrc: float, z: float, N: int, sparse_tol: float) -> torch.Tensor:
    modes = torch.zeros((sim_params.weights.shape[0], N, sim_params.Ny, sim_params.Nx), dtype=sim_params.dtype, device=sim_params.device)
    evals_tensor = torch.zeros((sim_params.weights.shape[0], N), dtype=sim_params.dtype, device=sim_params.device)

    for i, lam in enumerate(sim_params.lams):
        J = circ_mutual_intensity_sparse(sim_params, lam, rsrc, z, sparse_tol)
        
        evals, evecs = matrix_free_eigsh(J, N)
        
        evals = evals / evals.max()
        modes[i] = evecs.reshape(N, sim_params.Ny, sim_params.Nx)
        evals_tensor[i] = evals

    modes = modes.transpose(0, 1)
    evals_tensor = evals_tensor.transpose(0, 1)
    return modes, evals_tensor