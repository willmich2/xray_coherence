import torch # type: ignore
import cupy as cp # type: ignore
import cupyx.scipy.sparse.linalg as linalg # type: ignore

def construct_CSD_diag_sparse(
    x: torch.Tensor, 
    bandwidth: int) -> torch.Tensor:
    """
    Ensemble-average outer product ⟨x xᵀ⟩, keeping only a ±bandwidth band.

    This version is memory-efficient by pre-allocating tensors.

    Parameters
    ----------
    x : (N, 1, Nx) or (N, Nx) tensor
        The ensemble of 1-D column vectors.
    bandwidth : int
        Half-bandwidth k; only entries with |i - j| <= k are stored.

    Returns
    -------
    torch.Tensor (sparse COO, shape (Nx, Nx))
        Banded covariance matrix.
    """
    x = x.squeeze(1)  # (N, Nx)
    N, Nx = x.shape
    device = x.device
    dtype = x.dtype

    nnz = Nx + 2 * (bandwidth * Nx - bandwidth * (bandwidth + 1) // 2)

    # 2. Pre-allocate tensors for indices and values
    # Using torch.int64 for indices is standard
    indices = torch.empty((2, nnz), dtype=torch.int64, device=device)
    values = torch.empty(nnz, dtype=dtype, device=device)

    # 3. Fill the pre-allocated tensors in a loop
    current_pos = 0

    for k in range(-bandwidth, bandwidth + 1):
        k_abs = abs(k)
        num_elems = Nx - k_abs

        if num_elems <= 0:
            continue

        if k >= 0:
            # v = ⟨x_i * x_{i+k}⟩
            v = (x[:, :Nx - k] * x[:, k:].conj()).mean(dim=0)
            r = torch.arange(Nx - k, device=device)
            c = r + k
        else:  # k < 0
            # v = ⟨x_i * x_{i-|k|}⟩
            v = (x[:, -k:] * x[:, :Nx + k].conj()).mean(dim=0)
            c = torch.arange(Nx + k, device=device)
            r = c - k  # r = c + abs(k)

        # Write the calculated diagonal into the appropriate slice
        indices[0, current_pos : current_pos + num_elems] = r
        indices[1, current_pos : current_pos + num_elems] = c
        values[current_pos : current_pos + num_elems] = v

        # Update the position pointer
        current_pos += num_elems

    print('Loop and fill done')

    return torch.sparse_coo_tensor(indices, values, (Nx, Nx))

def matrix_free_eigsh(
    CSD_torch: torch.Tensor, 
    k: int):
    """
    Computes top k eigenvalues using a true matrix-free approach.
    It defines a LinearOperator for the mat-vec product without ever
    converting or duplicating the original sparse matrix.
    """
    CSD_torch = CSD_torch.coalesce()
    N = CSD_torch.shape[0]
    device = CSD_torch.device
    torch_dtype = CSD_torch.dtype
    # Convert the PyTorch dtype to its NumPy equivalent.
    # This is a robust way to handle the conversion for any data type.
    dtype = torch.tensor([], dtype=torch_dtype).numpy().dtype

    # 1. Define the matrix-vector product function (matvec)
    # This function will be called by the eigsh solver.
    def matvec(v_cupy):
        # The solver will pass a CuPy array. Convert it to a PyTorch tensor (zero-copy).
        v_torch = torch.as_tensor(v_cupy, device=device)

        # Perform the mat-vec product using PyTorch's sparse multiplication
        # The input vector might be 1D, so we unsqueeze/squeeze for matrix multiplication.
        result_torch = torch.sparse.mm(CSD_torch, v_torch.unsqueeze(1)).squeeze(1)

        # The solver expects a CuPy array back. Convert the result (zero-copy).
        return cp.asarray(result_torch)

    # 2. Create the LinearOperator
    # This object represents the matrix implicitly through its matvec action.
    lop = linalg.LinearOperator(shape=(N, N), matvec=matvec, dtype=dtype)

    # 3. Call the eigensolver with the LinearOperator
    evals, evecs = linalg.eigsh(lop, k, which='LA', tol=1e-4)

    # 4. Convert final results to PyTorch tensors
    evals_torch = torch.as_tensor(evals, device=device)
    evecs_torch = torch.as_tensor(evecs, device=device)

    return evals_torch, evecs_torch.transpose(0, 1)


def sparse_equal(a: torch.Tensor,
                 b: torch.Tensor,
                 *,
                 atol: float = 0.0,
                 rtol: float = 0.0) -> bool:
    """
    Return True when two `torch.sparse_coo_tensor` objects store the same data.

    Parameters
    ----------
    a, b : torch.Tensor
        Sparse COO tensors to compare.
    atol, rtol : float, optional
        Absolute and relative tolerances passed to `torch.allclose`
        when comparing the value arrays.

    Notes
    -----
    *  The tensors are first **coalesced** so that duplicate indices are summed
       and the index lists are sorted.  This produces a canonical ordering
       and guarantees that `.indices()` can be compared directly.:contentReference[oaicite:0]{index=0}
    *  After coalescing, two tensors are equal precisely when they have
       identical shapes, identical index arrays, and value arrays that are
       element-wise equal within the chosen tolerance.  The index arrays
       can be accessed only on a coalesced tensor.:contentReference[oaicite:1]{index=1}
    """
    # 1. Same shape?
    if a.shape != b.shape:
        return False

    # 2. Same sparse layout?
    if not (a.layout == torch.sparse_coo and b.layout == torch.sparse_coo):
        raise ValueError("Both inputs must be sparse COO tensors.")

    # 3. Canonicalise both tensors (sort + sum duplicates)
    a_c = a.coalesce()
    b_c = b.coalesce()

    # 4. Same number of non-zero entries?
    if a_c._nnz() != b_c._nnz():
        return False

    # 5. Compare index lists and values
    same_indices = torch.equal(a_c.indices(), b_c.indices())
    same_values  = torch.allclose(a_c.values(), b_c.values(),
                                  atol=atol, rtol=rtol)

    return same_indices and same_values