import torch # type: ignore
import torch.nn.functional as F # type: ignore
from src.simparams import SimParams

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
    del U
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
    del KX
    del KY

    # Let torch.sqrt handle the complex argument. This correctly models both
    # propagating waves (real sqrt) and evanescent waves (imaginary sqrt).
    kz = torch.sqrt(sqrt_arg)
    del sqrt_arg
    
    # The transfer function is now a batch of functions, one for each wavelength.
    # Shape: (batch, Ny_padded, Nx_padded)
    transfer_function = torch.exp(1j * z * kz)
    del kz
    # --- Apply Propagation in Fourier Domain ---
    # torch.fft.fft2 and ifft2 operate on the last two dimensions by default,
    # correctly handling the batch dimension.
    if U_padded.shape[1] == 1:
        U_padded_squeezed = U_padded.squeeze()
        del U_padded

        transfer_function_squeezed = transfer_function.squeeze()
        del transfer_function

        U_fourier = torch.fft.fft(U_padded_squeezed)
        del U_padded_squeezed

        U_z_padded = torch.fft.ifft(U_fourier * transfer_function_squeezed)
        del U_fourier
        del transfer_function_squeezed
        
        U_z_padded = U_z_padded.unsqueeze(1)
    else:
        U_fourier = torch.fft.fft2(U_padded)
        del U_padded
        U_z_padded = torch.fft.ifft2(U_fourier * transfer_function)
        del U_fourier
        del transfer_function

    # Unpad the result to the original spatial dimensions.
    U_z = unpad_half_width(U_z_padded)
    del U_z_padded
    
    # Return the full complex field to preserve phase information.
    return U_z


def direct_integration_propagation(
    U: torch.Tensor,
    lam: torch.Tensor,
    z: float,
    dx: float,
    device: torch.device
) -> torch.Tensor:
    """
    Performs wave propagation using direct numerical integration of the
    first Rayleigh-Sommerfeld diffraction integral.

    This method is computationally intensive (O(N^4) for an N*N grid) and is
    best suited for small grids or for validating results from faster methods
    like the angular spectrum method.

    Args:
        U (torch.Tensor): Input field, shape (batch, Ny, Nx). Can be real or complex.
        lam (torch.Tensor): Wavelength for each field in the batch, shape (batch,).
        z (float): Propagation distance. Must be non-zero.
        dx (float): Pixel size in the source and destination planes.
        device (torch.device): The torch device to use for calculations.

    Returns:
        torch.Tensor: The propagated complex field, shape (batch, Ny, Nx).
    """
    # --- Basic Setup ---
    # Ensure the input field is complex for the calculations.
    if not torch.is_complex(U):
        U = U.to(torch.complex64)

    batch_size, Ny, Nx = U.shape
    pi = torch.acos(torch.tensor(-1.0, dtype=torch.float32, device=device))

    # --- Setup Coordinate Grids ---
    # Create coordinates for the destination plane (x, y).
    # Using (N-1)/2 ensures the grid is centered at 0 for both odd and even N.
    x_coords = (torch.arange(Nx, device=device, dtype=torch.float32) - (Nx - 1) / 2) * dx
    y_coords = (torch.arange(Ny, device=device, dtype=torch.float32) - (Ny - 1) / 2) * dx
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # The source plane coordinates (xp, yp) are on the same grid.
    xp_coords = x_coords
    yp_coords = y_coords

    # --- Prepare Constants for Batched Operation ---
    # Reshape lam and calculate k for broadcasting across the spatial grid.
    # Shapes will be (batch, 1, 1).
    lam_reshaped = lam.view(batch_size, 1, 1).to(torch.float32)
    k = 2 * pi / lam_reshaped

    # The area of each source element for the integral.
    dx2 = dx**2

    # Initialize the output field.
    U_z = torch.zeros_like(U, dtype=torch.complex64)

    # --- Perform Direct Integration (Summation) ---
    # Iterate over each source point (p, q) and sum its contribution to the
    # entire destination field.
    for p in range(Ny):  # y' index
        for q in range(Nx):  # x' index
            # Get the field value at the source point for the whole batch.
            # Reshape to (batch, 1, 1) for broadcasting.
            U_source = U[:, p, q].view(batch_size, 1, 1)

            # Optimization: if the source point is zero across the entire batch,
            # it has no contribution, so we can skip it.
            if torch.all(U_source == 0):
                continue

            # Get the physical coordinates of the current source point.
            xp = xp_coords[q]
            yp = yp_coords[p]

            # --- Calculate the Impulse Response (Kernel) ---
            # Calculate the distance 'r' from the current source point (xp, yp)
            # to all points in the destination grid (X, Y).
            r_sq = (X - xp)**2 + (Y - yp)**2 + z**2
            r = torch.sqrt(r_sq)
            
            # The impulse response h(r) for the first Rayleigh-Sommerfeld integral is:
            # h(r) = (1 / (2*pi)) * (z/r) * (1/r - j*k) * (exp(j*k*r) / r)
            # This is an exact solution without paraxial approximations.

            # Calculate each term of the impulse response.
            # Broadcasting applies k (batch, 1, 1) to r (Ny, Nx).
            exp_term = torch.exp(1j * k * r)
            jk_over_r = 1j * k / r
            one_over_r_sq = 1 / r_sq
            
            # Combine terms to get the batched impulse response.
            # Shape: (batch, Ny, Nx)
            h_b = (1 / (2 * pi)) * (z / r) * (one_over_r_sq - jk_over_r) * exp_term

            # --- Sum the Contribution ---
            # Add the contribution of this source point to the total destination field.
            # This is the discrete version of the convolution integral.
            # U_source (batch,1,1) * h_b (batch,Ny,Nx) * dx2
            contribution = U_source * h_b * dx2
            U_z += contribution

    return U_z

def propagate_z(
    U: torch.Tensor,
    z: float,
    sim_params: SimParams, 
    method: str = "angular"
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
    if method == "angular":
        Uz = angular_spectrum_propagation(
            U,
            sim_params.lams,
            z,
            sim_params.dx,
            sim_params.device
        )
    elif method == "direct":
        Uz = direct_integration_propagation(
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
