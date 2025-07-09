import torch # type: ignore
import numpy as np # type: ignore

def create_objective_function(beta, forward_model, sim_params, forward_model_args):
    def objective_function(x, grad):
        """
        x:    current guess (NumPy array)
        grad: array for gradient output
        """
        # Convert to PyTorch tensor
        g = torch.tensor(x, dtype=torch.float32, requires_grad=True)

        # Apply smooth threshold
        g_thresholded = heaviside_projection(g, beta=beta)

        # Evaluate forward model
        obj = forward_model(g_thresholded, sim_params, *forward_model_args)

        # If NLopt wants gradients:
        if grad.size > 0:
            # Backprop in PyTorch
            obj.backward()
            # Copy gradients back to NLopt array
            grad[:] = g.grad.detach().numpy()

        return obj.item()
    return objective_function


def heaviside_projection(x, beta=10.0, eta=0.5):
    """
    Projects continuous x in [0,1] (approximately) 
    into near-binary values using a smooth approximation of a step function.
    """    
    numerator = torch.tanh(beta * (x - eta)) + np.tanh(beta * eta)
    denominator = np.tanh(beta * (1 - eta)) + np.tanh(beta * eta)
    return numerator / denominator