import torch # type: ignore
import torch.nn.functional as F # type: ignore


def angular_spectrum_propagation(U, lam, z, dx, device):
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

def propagate_z(u_init, z, params):
    Uz = torch.zeros((params.Ny, params.Nx), dtype=torch.complex64, device=params.device)
    
    for weight, lam in zip(params.weights, params.lams):
        Uz_lam = angular_spectrum_propagation(u_init, lam, z, params.dx, params.device)
        Uz += Uz_lam*weight
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


def apply_element(U, element, params):
    U_f = torch.zeros((params.Ny, params.Nx), dtype=torch.complex64, device=params.device)

    for lam in params.lams:
        U_lam = torch.zeros((params.Ny, params.Nx), dtype=torch.complex64, device=params.device)
        transmission = element.transmission(lam, element.n_elem, element.n_gap, params)
        U *= transmission
        U_f += U_lam
    return U_f


    