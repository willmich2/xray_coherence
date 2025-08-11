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
    Computes a sparse representation of the mutual intensity function using a
    memory-efficient, diagonal-by-diagonal vectorized approach.

    This function calculates the bandwidth of significant values, pre-allocates
    storage for the non-zero elements, and then computes each diagonal of the
    band in a vectorized manner, directly filling the sparse tensor arrays.

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
    k_wave = 2 * torch.pi / lam

    # --- 2. Determine Normalization Factor and Sparsity Threshold ---
    arg_at_min_dx = (dx_step * k_wave * r / z).cpu().numpy()
    
    if arg_at_min_dx < 1e-6:
        max_abs_val = 0.5
    else:
        max_abs_val = np.abs(scipy.special.jv(1, arg_at_min_dx) / arg_at_min_dx)
    
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
    
    dx_max = arg_cutoff * z / (k_wave * r)
    bandwidth = int(torch.ceil(dx_max / dx_step).item())

    # --- 4. Pre-allocate Tensors for Sparse Matrix Construction ---
    # Calculate the exact number of non-zero elements
    nnz = N + 2 * (bandwidth * N - bandwidth * (bandwidth + 1) // 2)
    
    indices = torch.empty((2, nnz), dtype=torch.long, device=device)
    values = torch.empty(nnz, dtype=torch.complex64, device=device)
    
    current_pos = 0

    # --- 5. Fill Tensors by Calculating Diagonals ---
    for k_diag in range(bandwidth + 1):
        num_elems = N - k_diag
        if num_elems <= 0:
            continue

        if k_diag == 0:
            # Main diagonal (k=0)
            rows = torch.arange(N, device=device)
            cols = rows
            vals = torch.ones(N, dtype=torch.complex64, device=device)
            
            indices[:, current_pos : current_pos + num_elems] = torch.stack([rows, cols])
            values[current_pos : current_pos + num_elems] = vals
            current_pos += num_elems
        else:
            # Off-diagonals (k > 0)
            # --- Upper diagonal (+k) ---
            rows_upper = torch.arange(num_elems, device=device)
            cols_upper = rows_upper + k_diag
            
            x1 = x[rows_upper]
            x2 = x[cols_upper]
            
            dx_vals = x2 - x1
            args = dx_vals * (k_wave * r / z)
            
            args_cpu = args.cpu().numpy()
            jinc_vals_cpu = np.divide(scipy.special.jv(1, args_cpu), args_cpu, where=args_cpu!=0)
            jinc_vals_cpu[args_cpu==0] = 0.5
            jinc_vals = torch.from_numpy(jinc_vals_cpu).to(device)
            
            psi = (torch.pi / (lam * z)) * (x2.pow(2) - x1.pow(2))
            vals_unnormalized = torch.exp(-1j * psi) * jinc_vals
            vals_normalized = vals_unnormalized / max_abs_val

            # Fill upper diagonal
            indices[:, current_pos : current_pos + num_elems] = torch.stack([rows_upper, cols_upper])
            values[current_pos : current_pos + num_elems] = vals_normalized
            current_pos += num_elems

            # --- Lower diagonal (-k) ---
            # Indices are swapped, values are conjugated (Hermitian)
            indices[:, current_pos : current_pos + num_elems] = torch.stack([cols_upper, rows_upper])
            values[current_pos : current_pos + num_elems] = vals_normalized.conj()
            current_pos += num_elems
            
    # --- 6. Create Final Sparse Tensor ---
    return torch.sparse_coo_tensor(indices, values, (N, N))


def incoherent_source(sim_params: SimParams, rsrc: float, z: float, N: int, sparse_tol: float) -> torch.Tensor:
    modes = torch.zeros((sim_params.weights.shape[0], N, sim_params.Ny, sim_params.Nx), dtype=sim_params.dtype, device=sim_params.device)
    evals_tensor = torch.zeros((sim_params.weights.shape[0], N), dtype=torch.float32, device=sim_params.device)

    for i, lam in enumerate(sim_params.lams):
        J = circ_mutual_intensity_sparse(sim_params, lam, rsrc, z, sparse_tol)
        
        evals, evecs = matrix_free_eigsh(J, N)
        
        evals = evals / evals.max()
        modes[i] = evecs.reshape(N, sim_params.Ny, sim_params.Nx)
        evals_tensor[i] = evals
        print(f"finished lam {i} of {sim_params.lams.shape[0]}")

    modes = modes.transpose(0, 1)
    evals_tensor = evals_tensor.transpose(0, 1)
    return modes, evals_tensor