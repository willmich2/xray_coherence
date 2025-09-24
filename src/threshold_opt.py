import numpy as np # type: ignore
import torch # type: ignore
import nlopt # type: ignore
from typing import Callable
from src.inversedesign_utils import create_objective_function, heaviside_projection
from src.simparams import SimParams
from src.forwardmodels import field_z_arbg_z, field_z_arbg_z_mode, forward_model_focus_plane_wave_power, forward_model_focus_incoherent_mc_power, propagate_z1_arbg_z2, forward_model_focus_incoherent_gaussian_schell_power, forward_model_focus_incoherent_psf_power, intensity_incoherent_psf, forward_model_focus_incoherent_psf_ratio
from src.elements import ArbitraryElement
from src.sources import gaussian_source
from src.montecarlo import mc_propagate_accumulate_intensity
from src.gaussian_schell import gaussian_schell_propagate_accumulate_intensity

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
    ) -> tuple[np.ndarray, list[float], list[np.ndarray]]:

    # compiled_model = torch.compile(forward_model, mode="default")
    
    # Initialize lists to track objective values and parameter vectors
    obj_values = []
    x_values = []

    for stage_idx, beta_val in enumerate(beta_schedule, 1):
        if print_results:
            print(f"\n--- Stage {stage_idx} with beta = {beta_val} ---")

        # Create NLopt optimizer
        n = x_init.shape[0]
        opt = nlopt.opt(method, n)
        
        # Create the base objective function for this stage
        base_obj_func = create_objective_function(
            beta=beta_val, 
            forward_model=forward_model, 
            sim_params=sim_params, 
            opt_params=opt_params, 
            forward_model_args=forward_model_args
            )
        
        # Create a wrapper that tracks objective values and parameter vectors
        def tracking_objective_function(x, grad):
            obj_val = base_obj_func(x, grad)
            obj_values.append(obj_val)
            x_values.append(x.copy())  # Make a copy to avoid reference issues
            return obj_val
        
        # Set the tracking objective function
        opt.set_max_objective(tracking_objective_function)
        
        # Optional bounds (could also be unbounded or partially bounded)
        opt.set_lower_bounds([0.0] * n)
        opt.set_upper_bounds([1.0] * n)

        opt.verbose = 1
        
        # Limit the number of function evaluations per stage
        opt.set_maxeval(max_eval_per_stage)

        opt.set_param("inner_maxeval", opt_params["inner_maxeval"])

        # Perform the optimization
        x_init = opt.optimize(x_init)

        final_obj = opt.last_optimum_value()

        if print_results:
            print(f"Stage {stage_idx} finished. obj = {(final_obj):.4f}")

    # convert to numpy arrays
    obj_values = np.array(obj_values)
    x_values = np.array(x_values)
    
    return x_init, obj_values, x_values


def x_I_opt(
        design_dict: dict, 
        ) -> tuple[np.ndarray, np.ndarray, float]:
    sim_params = design_dict["sim_params"]
    elem_params = design_dict["elem_params"]
    opt_params = design_dict["opt_params"]
    args = design_dict["args"]

    x_init = opt_params["x_init"]
    fwd_model = opt_params["forward_model"]
    
    method = opt_params["method"]
    betas = opt_params["betas"]
    max_eval = opt_params["max_eval"]
    
    opt_x, obj_values, x_values = threshold_opt(
        sim_params = sim_params, 
        opt_params = opt_params,
        forward_model = fwd_model, 
        forward_model_args = args, 
        beta_schedule = betas, 
        max_eval_per_stage = max_eval, 
        method = method,
        x_init = x_init, 
        print_results = False
        )

    final_obj = fwd_model(torch.tensor(opt_x), sim_params, opt_params, *args).cpu().numpy()
    
    opt_x_proj = heaviside_projection(torch.tensor(opt_x), beta = np.inf, eta = 0.5)
    
    init_params = SimParams(
        Ny=1, 
        Nx=x_init.shape[0]*2, 
        dx=sim_params.dx*opt_params["n"],
        device=sim_params.device, 
        dtype=sim_params.dtype,
        lams=sim_params.lams, 
        weights=sim_params.weights
    )
    
    opt_x_full = torch.repeat_interleave(
        torch.cat((opt_x_proj, torch.flip(opt_x_proj, dims=(0,)))).reshape(1, init_params.Nx), 
        opt_params["n"], 
        dim=1
        )
    
    # pad opt_x_full to (Ny, Nx)
    pad_left = (sim_params.Nx - opt_x_full.shape[1]) // 2
    pad_right = sim_params.Nx - opt_x_full.shape[1] - pad_left
    opt_x_full = torch.nn.functional.pad(opt_x_full, (pad_left, pad_right, 0, 0))

    # Use a generic hook on the forward model to compute intensity without branching.
    compute_intensity = getattr(fwd_model, "compute_intensity", None)
    if compute_intensity is None:
        raise ValueError("Selected forward model does not provide compute_intensity(x_full, sim_params, elem_params, args)")

    I_opt = compute_intensity(opt_x_full, sim_params, elem_params, args).reshape(sim_params.Nx).detach().cpu().numpy()
        
    opt_x = opt_x_proj.detach().cpu().numpy()
    opt_x_full = opt_x_full.detach().cpu().numpy()
            
    return opt_x, opt_x_full, I_opt, final_obj, obj_values, x_values