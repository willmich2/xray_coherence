import torch # type: ignore
import numpy as np # type: ignore
from src.propagation import propagate_z, apply_element
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
    Uzg = apply_element(Uz, element, sim_params)
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
        n_elem=elem_params["n_elem"], 
        n_gap=elem_params["n_gap"], 
        x=x
    )

    U_out_mc = mc_propagate(
        u_init_func=plane_wave, 
        u_init_func_args=(), # no arguments for plane wave
        prop_func=propagate_z_arbg_z,
        prop_func_args=(element,),
        n=1,
        z=z, 
        sim_params=sim_params
        )

    return U_out_mc

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
    U_out_mc = field_z_arbg_z(x_opt, sim_params, elem_params, z)

    I_out = U_out_mc.abs().pow(2).reshape(sim_params.Nx)
    
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
    x_dbl = torch.cat((x, torch.flip(x, dims=(0,))))
    n = opt_params["n"]
    x_opt = torch.repeat_interleave(x_dbl, n)

    U_out_mc = field_z_arbg_z(x_opt, sim_params, elem_params, z)

    I_out = U_out_mc.abs().pow(2).reshape(sim_params.Nx)
    
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
    U_out_mc = field_z_arbg_z(x_opt, sim_params, elem_params, z)

    I_out = U_out_mc.abs().pow(2).reshape(sim_params.Nx)

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
    it to the source intensity distribution for a computationally efficient result.

    Args:
        source_intensity (torch.Tensor): A 2D tensor representing the intensity
                                          distribution of the source.
        element_transmittance (torch.Tensor): A 2D tensor for the complex
                                              transmittance of the diffractive element.
        z (float): The propagation distance before and after the element.
        k (float): The wavenumber of the light (2 * pi / lambda).
        dx (float): The pixel size (sampling interval) in the spatial domain.
        propagate_field (callable): A function that propagates a complex field.
                                    Signature: propagate_field(field, z, k, dx) -> propagated_field
        apply_element (callable): A function that applies the diffractive element.
                                  Signature: apply_element(field, element) -> resulting_field

    Returns:
        torch.Tensor: A 2D tensor representing the final intensity at the output plane.
    """

    device = sim_params.device
    Ny, Nx = sim_params.Ny, sim_params.Nx

    element = ArbitraryElement(
        name="ArbitraryElement", 
        thickness=elem_params["thickness"], 
        n_elem=elem_params["n_elem"], 
        n_gap=elem_params["n_gap"], 
        x=x
    )

    point_source_field = torch.zeros((Ny, Nx), dtype=torch.complex64, device=device)
    point_source_field[Ny // 2, Nx // 2] = 1.0

    h_sys = propagate_z_arbg_z(
        U = point_source_field, 
        z = z, 
        sim_params = sim_params, 
        element = element
        )

    otf_sys = torch.fft.fft2(torch.fft.ifftshift(torch.abs(h_sys)** 2))

    source_intensity_ft = torch.fft.fft2(torch.fft.ifftshift(torch.abs(gaussian_source(sim_params, rsrc))**2))
    final_intensity_ft = source_intensity_ft * otf_sys

    final_intensity = torch.fft.fftshift(torch.fft.ifft2(final_intensity_ft))

    # intensity must be real. small imaginary parts may exist due to numerical error.
    return final_intensity.real


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


