import torch # type: ignore
import numpy as np # type: ignore
from src.propagation import propagate_z, angular_spectrum_propagation
from src.sources import plane_wave, gaussian_source
from src.elements import ArbitraryElement
from src.simparams import SimParams
from src.montecarlo import mc_propagate

# TODO: 
# - [] probably get rid of mc_propagate since we are modeling incoherence with the OTF
# - [] add a function to calculate the OTF


def propagate_z_arbg_z(
    U: torch.Tensor, 
    z: float, 
    sim_params: SimParams, 
    element: ArbitraryElement
    ) -> torch.Tensor:
    """
    Propagate a plane wave a distance z, apply an arbitrary element, and propagate a distance z again.
    """
    Uz = propagate_z(U, z, sim_params)
    Uzg = element.apply_element(Uz, sim_params)
    Uzgz = propagate_z(Uzg, z, sim_params)
    return Uzgz

def field_z_arbg_z(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    z: float, 
    ) -> torch.Tensor:
    """
    Propagate a plane wave a distance z, apply an arbitrary element, and propagate a distance z again.
    """
    element = ArbitraryElement(
        name="ArbitraryElement", 
        thickness=elem_params["thickness"], 
        elem_map=elem_params["elem_map"], 
        gap_map=elem_params["gap_map"], 
        x=x
    )

    Uzgz = propagate_z_arbg_z(
        U = plane_wave(sim_params), 
        z = z, 
        sim_params = sim_params, 
        element = element
        )

    return Uzgz

def forward_model_focus_plane_wave(
    x: torch.Tensor, 
    sim_params: SimParams,
    opt_params: dict,
    elem_params: dict,
    Ncenter: int,
    Navg: int,
    z: float, 
    ) -> float:
    """
    Propagate a plane wave a distance z, apply an arbitrary element, and propagate a distance z again.
    Then, calculate the visibility of the output field.
    """
    n = opt_params["n"]
    x_opt = torch.repeat_interleave(x, n)
    Uzgz = field_z_arbg_z(x_opt, sim_params, elem_params, z)

    I_out = torch.sum(Uzgz.abs().pow(2) * sim_params.weights, dim=0).reshape(sim_params.Nx)
    
    I_out_center = I_out[I_out.shape[0]//2 - Ncenter//2:I_out.shape[0]//2 + Ncenter//2].mean()
    I_out_edge = torch.cat((I_out[:Navg], I_out[-Navg:])).mean()
    
    visibility = (I_out_center - I_out_edge) / (I_out_center + I_out_edge)

    obj = visibility

    return obj

def forward_model_focus_plane_wave_power(
    x: torch.Tensor, 
    sim_params: SimParams,
    opt_params: dict,
    elem_params: dict,
    Ncenter: int,
    z: float, 
    ) -> float:
    """
    Propagate a plane wave a distance z, apply an arbitrary element, and propagate a distance z again.
    Then, calculate the power within a center region of the output field.
    """
    # concatenate x and a backwards version of x
    x_dbl = torch.cat((x, x[torch.arange(x.numel() - 1, -1, -1)]))
    n = opt_params["n"]
    x_opt = torch.repeat_interleave(x_dbl, n)

    Uzgz = field_z_arbg_z(x_opt, sim_params, elem_params, z)

    # calculated intensity by summing over wavelengths, weighted by weights
    weights_t = sim_params.weights.view(-1, 1, 1)
    I_out = torch.sum(Uzgz.abs().pow(2) * weights_t, dim=0).reshape(sim_params.Nx)
    
    P_out_center = I_out[I_out.shape[0]//2 - Ncenter//2:I_out.shape[0]//2 + Ncenter//2].sum()

    obj = P_out_center

    return obj

def forward_model_focus_plane_wave_overlap(
    x: torch.Tensor, 
    sim_params: SimParams,
    opt_params: dict,
    elem_params: dict,
    r_focus: float,
    z: float, 
    ) -> float:
    """
    Propagate a plane wave a distance z, apply an arbitrary element, and propagate a distance z again.
    Then, calculate the power within a center region of the output field.
    """
    n = opt_params["n"]
    x_opt = torch.repeat_interleave(x, n)
    Uzgz = field_z_arbg_z(x_opt, sim_params, elem_params, z)

    I_out = torch.sum(Uzgz.abs().pow(2) * sim_params.weights, dim=0).reshape(sim_params.Nx)

    R = torch.sqrt(sim_params.X**2 + sim_params.Y**2)
    I_focus = torch.exp(-R**2 / (2 * r_focus**2))

    obj = torch.sum(I_out * I_focus) / torch.sum(I_focus) / torch.sum(I_out)

    return obj

def propagate_z_arbg_z_incoherent(
    x: torch.Tensor, 
    sim_params: SimParams,
    elem_params: dict,
    z: float, 
    rsrc: float
) -> torch.Tensor:
    """
    Calculates the final intensity of an incoherent source passing through a system.

    This function computes the system's Optical Transfer Function (OTF) and applies
    it to the source intensity distribution.
    """

    device = sim_params.device
    Ny, Nx = sim_params.Ny, sim_params.Nx

    element = ArbitraryElement(
        name="ArbitraryElement", 
        thickness=elem_params["thickness"], 
        elem_map=elem_params["elem_map"], 
        gap_map=elem_params["gap_map"], 
        x=x
    )

    point_source_field = torch.zeros((Ny, Nx), dtype=sim_params.dtype, device=device)
    point_source_field[Ny // 2, Nx // 2] = 1.0

    I_f = torch.zeros((Ny, Nx), dtype=point_source_field.real.dtype, device=device)

    for weight, lam in zip(sim_params.weights, sim_params.lams):

        Uz_lam = angular_spectrum_propagation(
            U = point_source_field, 
            lam = lam, 
            z = z, 
            dx = sim_params.dx, 
            device = device
            )
        
        t_lam = element.transmission(lam, sim_params)
        Uzg_lam = Uz_lam * t_lam

        h_lam = angular_spectrum_propagation(
            U = Uzg_lam, 
            lam = lam, 
            z = z, 
            dx = sim_params.dx, 
            device = device
            )
        
        otf_lam = torch.fft.fft2(torch.fft.ifftshift(torch.abs(h_lam)**2))

        I_src_lam_ft = torch.fft.fft2(torch.fft.ifftshift(torch.abs(gaussian_source(sim_params, rsrc))**2))
        I_f_lam_ft = I_src_lam_ft * otf_lam

        I_f_lam = torch.fft.fftshift(torch.fft.ifft2(I_f_lam_ft))

        I_f += weight * I_f_lam

    # intensity must be real. small imaginary parts may exist due to numerical error.
    return I_f.real


def focus_incoherent_power(
    x: torch.Tensor, 
    sim_params: SimParams,
    opt_params: dict,
    elem_params: dict,
    Ncenter: int,
    z: float, 
    rsrc: float
    ) -> float:
    """
    Propagate a plane wave a distance z, apply an arbitrary element, and propagate a distance z again.
    Then, calculate the power within a center region of the output field.
    """
    x_dbl = torch.cat((x, torch.flip(x, dims=(0,))))
    n = opt_params["n"]
    x_opt = torch.repeat_interleave(x_dbl, n)

    I_out = propagate_z_arbg_z_incoherent(x_opt, sim_params, elem_params, z, rsrc)

    P_out_center = I_out[I_out.shape[0]//2 - Ncenter//2:I_out.shape[0]//2 + Ncenter//2].sum()

    obj = P_out_center

    return obj


