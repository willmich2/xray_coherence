import torch
from propagation import angular_spectrum_propagation


def mc_propagate(func, n, u_init_func, z, params):
    mc_tensor = torch.zeros((n, params.Ny, params.Nx), dtype=torch.complex64, device=params.device)

    for i in range(n):
        u_init = u_init_func()
        u_final = func(u_init, z, params)

        mc_tensor[i, :, :] = u_final
    return mc_tensor