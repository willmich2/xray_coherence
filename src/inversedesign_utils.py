import torch # type: ignore
import numpy as np # type: ignore
from typing import Callable
from src.simparams import SimParams
from src.elements import ZonePlate
import copy
import torch.nn.functional as F # type: ignore

def create_objective_function(
    beta: float, 
    forward_model: Callable, 
    sim_params: SimParams, 
    opt_params: dict, 
    forward_model_args: tuple
    ) -> Callable:
    def objective_function(x, grad):
        """
        x:    current guess (NumPy array)
        grad: array for gradient output
        """
        zero = torch.zeros(0, dtype=sim_params.dtype, device=sim_params.device)
        # Convert to PyTorch tensor
        g = torch.tensor(x, dtype=zero.real.dtype, requires_grad=True)

        if g.dim() == 1:
            g = g.unsqueeze(0).unsqueeze(0)
        elif g.dim() == 2:
            g = g.unsqueeze(1) # Assume (batch, length) -> (batch, channels, length)

        # apply density filtering
        g_filtered = density_filtering(g, opt_params["filter_radius"], sim_params)

        # Apply smooth threshold
        g_thresholded = heaviside_projection(g_filtered, beta=beta)

        # enforce feature size
        g_physical = feature_size_filtering(g_thresholded, opt_params["min_feature_radius"], sim_params)
        g_physical = g_physical.squeeze(0).squeeze(0)

        # Evaluate forward model
        obj = forward_model(g_physical, sim_params, opt_params, *forward_model_args)

        # If NLopt wants gradients:
        if grad.size > 0:
            # Backprop in PyTorch
            obj.backward()
            # Copy gradients back to NLopt array
            grad[:] = g.grad.detach().numpy()

        return obj.item()
    return objective_function


def density_filtering(
    x: torch.Tensor,
    filter_radius: float,
    sim_params: SimParams
) -> torch.Tensor:

    fiter_radius_int = int(filter_radius / sim_params.dx)
    kernel_size_filter = 2 * fiter_radius_int + 1
    # Create a 1D cone filter kernel
    cone_kernel = torch.tensor([1.0 - abs(i - fiter_radius_int) / (fiter_radius_int + 1)
                                for i in range(kernel_size_filter)],
                               device=x.device, dtype=x.dtype, requires_grad=False)
    cone_kernel = cone_kernel.view(1, 1, -1) / cone_kernel.sum()
    # Apply convolution
    x_filtered = F.conv1d(x, cone_kernel, padding='same')
    return x_filtered


def feature_size_filtering(
    x: torch.Tensor,
    min_feature_radius: float,
    sim_params: SimParams
) -> torch.Tensor:

    min_feature_radius_int = int(min_feature_radius / sim_params.dx)    
    kernel_size_morph = 2 * min_feature_radius_int + 1
    padding_morph = min_feature_radius_int

    # Use a single max pooling operation instead of erosion+dilation
    # This is more gradient-friendly and still enforces minimum feature size
    x_physical = F.max_pool1d(x,
                             kernel_size=kernel_size_morph,
                             stride=1,
                             padding=padding_morph)

    return x_physical


def heaviside_projection(
    x: torch.Tensor, 
    beta: float = 10.0, 
    eta: float = 0.5
    ) -> torch.Tensor:
    """
    Projects continuous x in [0,1] (approximately) 
    into near-binary values using a smooth approximation of a step function.
    """    
    numerator = torch.tanh(beta * (x - eta)) + torch.tanh(torch.tensor(beta * eta, device=x.device, dtype=x.dtype, requires_grad=False))
    denominator = torch.tanh(torch.tensor(beta * (1 - eta), device=x.device, dtype=x.dtype, requires_grad=False)) + torch.tanh(torch.tensor(beta * eta, device=x.device, dtype=x.dtype, requires_grad=False))
    return numerator / denominator


def zp_init(
        lam: float, 
        f: float, 
        min_feature_size: float, 
        sim_params: SimParams, 
        opt_params: dict
) -> np.ndarray:
    zone_plate = ZonePlate(
        name = "zp_init", 
        thickness = 1, 
        f = f,
        min_feature_size = min_feature_size, 
        elem_map = [np.array([0, np.inf]), np.array([1., 1.])], 
        gap_map = [np.array([0, np.inf]), np.array([1 + 1j*np.inf, 1 + 1j*np.inf])]
    )

    zp_trans = zone_plate.transmission(lam, lam, sim_params).abs()
    zp_init = torch.where(zp_trans > 0.5, 1.0, 0.0).cpu().reshape(sim_params.Nx)[::opt_params["n"]]
    zp_init = zp_init[:zp_init.shape[0]//2].numpy()
    return zp_init


def get_iterative_wavelength_design_dicts(design_dict):
    lams, weights = design_dict["sim_params"].lams, design_dict["sim_params"].weights
    Nwvl = lams.shape[0]
    design_dicts = []

    for i in range(Nwvl):
        design_dict_i = copy.deepcopy(design_dict)
        # choose central i+1 wavelengths
        slc = slice(Nwvl // 2 - (i+1)//2, Nwvl // 2 + (i+2)//2)
        lams_i = lams[slc]
        weights_i = weights[slc]
        design_dict_i["sim_params"].lams = lams_i
        design_dict_i["sim_params"].weights = weights_i
        design_dicts.append(design_dict_i)
    return design_dicts