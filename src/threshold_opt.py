import torch # type: ignore
import numpy as np # type: ignore
import nlopt # type: ignore
from src.inversedesign_utils import create_objective_function # type: ignore
from src.simparams import SimParams
from typing import Callable

def threshold_opt(
    sim_params: SimParams, 
    opt_params: dict,
    forward_model: Callable,
    forward_model_args: tuple,
    beta_schedule: list[float], 
    max_eval_per_stage: int, 
    method,
    x_init: np.ndarray, 
    print_results: bool = True
    ) -> np.ndarray:

    compiled_model = torch.compile(forward_model, mode="max-autotune")

    for stage_idx, beta_val in enumerate(beta_schedule, 1):
        if print_results:
            print(f"\n--- Stage {stage_idx} with beta = {beta_val} ---")

        # Create NLopt optimizer
        n = int(sim_params.Nx // opt_params["n"] // 2)
        opt = nlopt.opt(method, n)
        
        # Set objective function for this stage, using the current beta
        opt.set_max_objective(create_objective_function(
            beta=beta_val, 
            forward_model=compiled_model, 
            sim_params=sim_params, 
            opt_params=opt_params, 
            forward_model_args=forward_model_args
            ))
        
        # Optional bounds (could also be unbounded or partially bounded)
        opt.set_lower_bounds([0.0] * n)
        opt.set_upper_bounds([1.0] * n)

        opt.verbose = 1
        
        # Limit the number of function evaluations per stage
        opt.set_maxeval(max_eval_per_stage)

        # Perform the optimization
        x_init = opt.optimize(x_init)

        final_obj = create_objective_function(
            beta=beta_val, 
            forward_model=forward_model, 
            sim_params=sim_params, 
            opt_params=opt_params, 
            forward_model_args=forward_model_args
            )(x_init, np.array([]))
        if print_results:
            print(f"Stage {stage_idx} finished. obj = {(final_obj):.4f}")
    
    threshold_obj = create_objective_function(
            beta=np.inf, 
            forward_model=forward_model, 
            sim_params=sim_params, 
            opt_params=opt_params, 
            forward_model_args=forward_model_args
            )(x_init, np.array([]))
    if print_results:
        print(f"Threshold obj = {(threshold_obj):.4f}")

    return x_init, threshold_obj