import torch # type: ignore
import numpy as np # type: ignore
from typing import Callable
from src.simparams import SimParams
from src.elements import ZonePlate

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
        try:
            zero = torch.zeros(0, dtype=sim_params.dtype, device=sim_params.device)
            # Convert to PyTorch tensor
            g = torch.tensor(x, dtype=zero.real.dtype, requires_grad=True)

            # Apply smooth threshold
            g_thresholded = heaviside_projection(g, beta=beta)

            # Evaluate forward model
            obj = forward_model(g_thresholded, sim_params, opt_params, *forward_model_args)

            # Check for invalid values
            if torch.isnan(obj) or torch.isinf(obj):
                print(f"Warning: Objective function returned {obj.item()}")
                return 0.0  # Return a safe default value

            # If NLopt wants gradients:
            if grad.size > 0:
                # Backprop in PyTorch
                obj.backward()
                # Copy gradients back to NLopt array
                grad[:] = g.grad.detach().numpy()
                
                # Check for invalid gradients
                if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                    print(f"Warning: Gradients contain invalid values")
                    grad[:] = 0.0  # Set gradients to zero

            return obj.item()
        except Exception as e:
            print(f"Error in objective function: {e}")
            # Return a safe default value
            if grad.size > 0:
                grad[:] = 0.0
            return 0.0
    return objective_function


def heaviside_projection(
    x: torch.Tensor, 
    beta: float = 10.0, 
    eta: float = 0.5
    ) -> torch.Tensor:
    """
    Projects continuous x in [0,1] (approximately) 
    into near-binary values using a smooth approximation of a step function.
    """    
    numerator = torch.tanh(beta * (x - eta)) + torch.tanh(torch.tensor(beta * eta, device=x.device, dtype=x.dtype, requires_grad=True))
    denominator = torch.tanh(torch.tensor(beta * (1 - eta), device=x.device, dtype=x.dtype, requires_grad=True)) + torch.tanh(torch.tensor(beta * eta, device=x.device, dtype=x.dtype, requires_grad=True))
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