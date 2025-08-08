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
    allocating the full dense matrix in memory.

    This function leverages the fact that the mutual intensity function's
    magnitude decays with distance, meaning the resulting matrix is effectively
    band-diagonal. It calculates the bandwidth of significant values and only
    computes elements within that band.

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
    k = 2 * np.pi / lam

    # --- 2. Determine Normalization Factor and Sparsity Threshold ---
    # The maximum absolute value of the off-diagonal elements is determined by the
    # jinc function J1(x)/x, and its peak is near the smallest spatial separation.
    arg_at_max = (dx_step * k * r / z).cpu().numpy()
    
    # The jinc function J1(y)/y approaches 0.5 as y->0.
    if arg_at_max < 1e-6:
        max_abs_val = 0.5
    else:
        max_abs_val = np.abs(scipy.special.jv(1, arg_at_max) / arg_at_max)
    
    # This is the absolute value threshold for the un-normalized jinc values.
    abs_threshold = sparse_tol * max_abs_val

    # --- 3. Calculate Bandwidth of Significant Elements ---
    # We need to find the argument `arg_cutoff` for the jinc function where its
    # value drops below our absolute threshold. We use a numerical solver for this.
    def func_to_solve(arg, target):
        # Function whose root we want to find: |jinc(arg)| - target = 0
        value = np.abs(scipy.special.jv(1, arg) / arg) if arg > 1e-9 else 0.5
        return value - target

    # Establish a search bracket [a, b] for the root finder.
    a = arg_at_max
    b = a + 1.0 
    # Expand the bracket until the function value at b is below the target.
    while func_to_solve(b, abs_threshold) > 0:
        b *= 2

    # Find the root using a robust numerical method (Brent's method).
    arg_cutoff = scipy.optimize.brentq(func_to_solve, a, b, args=(abs_threshold,))
    
    # Convert the argument cutoff to a maximum spatial distance and then an index bandwidth.
    dx_max = arg_cutoff * z / (k * r)
    bandwidth = int(np.ceil(dx_max.cpu().numpy() / dx_step.cpu().numpy()))

    # --- 4. Build Sparse Matrix Elements in COO Format ---
    rows, cols, vals = [], [], []
    x_cpu = x.cpu().numpy() # Use numpy for faster looping and scipy compatibility

    for i in range(N):
        # The diagonal element is always 1.
        rows.append(i)
        cols.append(i)
        vals.append(torch.tensor(1.0 + 0.0j, dtype=torch.complex64, device=device))

        # Iterate only over the upper band for this row.
        for j in range(i + 1, min(N, i + bandwidth + 1)):
            dx_val = x_cpu[j] - x_cpu[i]
            arg = dx_val * k * r / z
            arg = arg.cpu().numpy()
            
            jinc_val = scipy.special.jv(1, arg) / arg if arg > 1e-9 else 0.5
            
            # This check is technically redundant if bandwidth is precise, but is good practice.
            if np.abs(jinc_val) >= abs_threshold:
                psi = (np.pi / (lam * z)) * (x_cpu[j]**2 - x_cpu[i]**2).cpu().numpy()
                val_unnormalized = np.exp(-1j * psi) * jinc_val
                
                # Normalize the value before adding it to the list.
                val_normalized = torch.tensor(val_unnormalized / max_abs_val, dtype=torch.complex64, device=device)
                
                # Add the (i, j) element.
                rows.append(i)
                cols.append(j)
                vals.append(val_normalized)

                # Add the Hermitian conjugate (j, i) element.
                rows.append(j)
                cols.append(i)
                vals.append(val_normalized.conj())

    # --- 5. Create Final Sparse Tensor ---
    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.stack(vals)
    
    # Create the sparse tensor and coalesce it to sum duplicates and sort indices.
    J_sparse = torch.sparse_coo_tensor(indices, values, (N, N))
    return J_sparse.coalesce()


def incoherent_source(sim_params: SimParams, rsrc: float, z: float, N: int, sparse_tol: float) -> torch.Tensor:
    modes = torch.zeros((sim_params.weights.shape[0], N, sim_params.Ny, sim_params.Nx), dtype=sim_params.dtype, device=sim_params.device)
    evals = torch.zeros((sim_params.weights.shape[0], N), dtype=sim_params.dtype, device=sim_params.device)

    for i, lam in enumerate(sim_params.lams):
      J = circ_mutual_intensity_sparse(sim_params, lam, rsrc, z, sparse_tol)
      
      evals, evecs = matrix_free_eigsh(J, N)
      
      evals = evals / evals.max()
      modes[i] = evecs
      evals[i] = evals

    modes = modes.transpose(0, 1)
    evals = evals.transpose(0, 1)
    return modes, evals