from src.simparams import SimParams
import torch # type: ignore
from typing import Callable

def mc_propagate(
    u_init_func: Callable, 
    u_init_func_args: tuple,
    prop_func: Callable, 
    prop_func_args: tuple,
    n: int, 
    z: float, 
    sim_params: SimParams
    ) -> torch.Tensor:
    """
    Repeatedly propagate an initial field given by u_init_func through a propagation function prop_func.
    Returns a tensor of shape (n, Ny, Nx) where n is the number of Monte Carlo samples.
    """
    mc_tensor = torch.zeros((n, sim_params.weights.shape[0], sim_params.Ny, sim_params.Nx), dtype=sim_params.dtype, device=sim_params.device)

    for i in range(n):
        u_init = u_init_func(sim_params, *u_init_func_args)
        u_final = prop_func(u_init, z, sim_params, *prop_func_args)

        mc_tensor[i, :, :, :] = u_final
    return mc_tensor


def mc_propagate_accumulate_intensity(
    u_init_func: Callable, 
    u_init_func_args: tuple, 
    prop_func: Callable, 
    prop_func_args: tuple, 
    n: int, 
    z: float, 
    sim_params: SimParams
    ) -> torch.Tensor:
    """
    Repeatedly propagate an initial field given by u_init_func through a propagation function prop_func.
    Returns a tensor of shape (Nwvl, Ny, Nx) where n is the number of Monte Carlo samples.
    """
    i_final = torch.zeros((sim_params.weights.shape[0], sim_params.Ny, sim_params.Nx), dtype=sim_params.dtype, device=sim_params.device)

    for i in range(n):
        u_init = u_init_func(sim_params, *u_init_func_args)
        u_final = prop_func(u_init, z, sim_params, *prop_func_args)
        i_final += torch.abs(u_final)**2

    i_final /= n
    
    return i_final