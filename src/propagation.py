import torch # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np # type: ignore
from src.simparams import SimParams
from src.util import refractive_index_at_wvl

def angular_spectrum_propagation(
    U: torch.Tensor,
    lam: torch.Tensor,
    z: float,
    dx: float,
    device: torch.device
) -> torch.Tensor:
    """
    Performs angular spectrum propagation for a batch of fields.

    Args:
        U (torch.Tensor): Input field, shape (batch, Ny, Nx). Can be real or complex.
        lam (torch.Tensor): Wavelength for each field in the batch, shape (batch,).
        z (float): Propagation distance.
        dx (float): Pixel size.
        device (torch.device): The torch device to use for calculations.

    Returns:
        torch.Tensor: The propagated complex field, shape (batch, Ny, Nx).
    """
    # Pad the input field to avoid aliasing from circular convolution.
    # Assumes pad_double_width correctly pads the last two (spatial) dimensions.
    U_padded = pad_double_width(U)
    batch_size, Ny_padded, Nx_padded = U_padded.shape

    # --- Setup constants and coordinates ---
    pi = torch.acos(torch.tensor(-1.0, dtype=torch.float32, device=device))

    # Reshape wavelengths to (batch, 1, 1) for broadcasting.
    # Using reshape() is more robust and avoids the TypeError.
    lam_reshaped = lam.reshape(batch_size, 1, 1).to(torch.float32)
    
    # Calculate wave number k0 for each wavelength in the batch.
    # Shape: (batch, 1, 1)
    k0 = 2 * pi / lam_reshaped

    # Create spatial frequency coordinates (these are the same for all items in batch).
    kx = torch.fft.fftfreq(Nx_padded, dx, dtype=torch.float32, device=device) * 2 * pi
    ky = torch.fft.fftfreq(Ny_padded, dx, dtype=torch.float32, device=device) * 2 * pi
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')

    # --- Construct the Transfer Function ---
    # The core calculation is now batched.
    # Broadcasting (batch, 1, 1) with (Ny_padded, Nx_padded) results in (batch, Ny_padded, Nx_padded).
    sqrt_arg = k0**2 - KX**2 - KY**2
    
    # Let torch.sqrt handle the complex argument. This correctly models both
    # propagating waves (real sqrt) and evanescent waves (imaginary sqrt).
    kz = torch.sqrt(sqrt_arg)
    
    # The transfer function is now a batch of functions, one for each wavelength.
    # Shape: (batch, Ny_padded, Nx_padded)
    transfer_function = torch.exp(1j * z * kz)

    # --- Apply Propagation in Fourier Domain ---
    # torch.fft.fft2 and ifft2 operate on the last two dimensions by default,
    # correctly handling the batch dimension.
    U_fourier = torch.fft.fft2(U_padded)
    U_z_padded = torch.fft.ifft2(U_fourier * transfer_function)

    # Unpad the result to the original spatial dimensions.
    U_z = unpad_half_width(U_z_padded)
    
    # Return the full complex field to preserve phase information.
    return U_z

def propagate_z(
    U: torch.Tensor,
    z: float,
    sim_params: "SimParams"
) -> torch.Tensor:
    """
    Propagates a multi-wavelength field U over a distance z.

    Args:
        U (torch.Tensor): Input field, shape (num_wavelengths, Ny, Nx).
        z (float): Propagation distance.
        sim_params (SimParams): Object containing simulation parameters like
                                wavelengths, pixel size, and device.

    Returns:
        torch.Tensor: The propagated complex field, shape (num_wavelengths, Ny, Nx).
    """
    # The for-loop is replaced with a single, batched call to the
    # modified angular_spectrum_propagation function. The first dimension of U
    # (num_wavelengths) is treated as the batch dimension.
    Uz = angular_spectrum_propagation(
        U,
        sim_params.lams,
        z,
        sim_params.dx,
        sim_params.device
    )
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
