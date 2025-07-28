import torch # type: ignore
import numpy as np # type: ignore
from typing import Callable
from src.simparams import SimParams

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

        # Apply smooth threshold
        g_thresholded = heaviside_projection(g, beta=beta)

        # Evaluate forward model
        obj = forward_model(g_thresholded, sim_params, opt_params, *forward_model_args)

        # If NLopt wants gradients:
        if grad.size > 0:
            # Backprop in PyTorch
            obj.backward()
            # Copy gradients back to NLopt array
            grad[:] = g.grad.detach().numpy()

        return obj.item()
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