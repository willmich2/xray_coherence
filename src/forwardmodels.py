import torch # type: ignore
import numpy as np # type: ignore
from src.propagation import propagate_z, apply_element
from src.sources import plane_wave
from src.elements import ArbitraryElement
from src.simparams import SimParams
from src.montecarlo import mc_propagate

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
    x: np.ndarray,
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
    x: np.ndarray, 
    sim_params: SimParams,
    elem_params: dict,
    forward_model_args: dict,
    z: float, 
    ) -> float:
    """
    Propagate a plane wave a distance z, apply an arbitrary element, and propagate a distance z again.
    Then, calculate the visibility of the output field.
    """
    U_out_mc = field_z_arbg_z(x, sim_params, elem_params, z)

    Ncenter = forward_model_args["Ncenter"]
    Navg = forward_model_args["Navg"]
    I_out = U_out_mc.abs().pow(2).reshape(sim_params.Nx)
    
    I_out_center = I_out[I_out.shape[0]//2 - Ncenter//2:I_out.shape[0]//2 + Ncenter//2].mean()
    I_out_edge = torch.cat((I_out[:Navg], I_out[-Navg:])).mean()
    
    visibility = (I_out_center - I_out_edge) / (I_out_center + I_out_edge)

    obj = visibility

    return obj

def forward_model_focus_plane_wave_power(
    x: torch.Tensor, 
    sim_params: SimParams,
    elem_params: dict,
    forward_model_args: dict,
    z: float, 
    ) -> float:
    """
    Propagate a plane wave a distance z, apply an arbitrary element, and propagate a distance z again.
    Then, calculate the power within a center region of the output field.
    """
    U_out_mc = field_z_arbg_z(x, sim_params, elem_params, z)

    Ncenter = forward_model_args["Ncenter"]

    I_out = U_out_mc.abs().pow(2).reshape(sim_params.Nx)
    
    P_out_center = I_out[I_out.shape[0]//2 - Ncenter//2:I_out.shape[0]//2 + Ncenter//2].sum()

    obj = P_out_center

    return obj