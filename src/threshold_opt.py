import torch # type: ignore
import numpy as np # type: ignore
import nlopt # type: ignore
from src.inversedesign_utils import create_objective_function # type: ignore
from src.simparams import SimParams
from typing import Callable

def threshold_opt(
    sim_params: SimParams, 
    forward_model: Callable,
    forward_model_args: tuple,
    beta_schedule: list[float], 
    max_eval_per_stage: int, 
    x_init: np.ndarray
    ) -> np.ndarray:
    
    for stage_idx, beta_val in enumerate(beta_schedule, 1):
        print(f"\n--- Stage {stage_idx} with beta = {beta_val} ---")

        # Create NLopt optimizer
        opt = nlopt.opt(nlopt.LD_MMA, sim_params.Nx)
        
        # Set objective function for this stage, using the current beta
        opt.set_max_objective(create_objective_function(beta_val, forward_model, sim_params, forward_model_args))
        
        # Optional bounds (could also be unbounded or partially bounded)
        opt.set_lower_bounds([0.0] * sim_params.Nx)
        opt.set_upper_bounds([1.0] * sim_params.Nx)

        opt.verbose = 1
        
        # Limit the number of function evaluations per stage
        opt.set_maxeval(max_eval_per_stage)

        # Perform the optimization
        x_init = opt.optimize(x_init)

        final_obj = create_objective_function(
            beta_val, 
            forward_model, 
            sim_params, 
            forward_model_args
            )(x_init, np.array([]))
        print(f"Stage {stage_idx} finished. obj = {(final_obj):.4f}")
    
    return x_init