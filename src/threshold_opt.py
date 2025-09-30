import numpy as np # type: ignore
import torch # type: ignore
import nlopt # type: ignore
from typing import Callable
from src.inversedesign_utils import create_objective_function, heaviside_projection, get_iterative_wavelength_design_dicts, density_filtering
from src.simparams import SimParams
import copy

def threshold_opt(
    design_dict: dict,
    print_results: bool = True  
    ) -> tuple[np.ndarray, list[float], list[np.ndarray]]:

    # compiled_model = torch.compile(forward_model, mode="default")
    
    # Initialize lists to track objective values and parameter vectors
    obj_values = []
    x_values = []

    # Unpack the design dictionary
    sim_params: SimParams = design_dict["sim_params"]
    opt_params: dict = design_dict["opt_params"]
    forward_model: Callable = opt_params["forward_model"]
    forward_model_args: tuple = design_dict["args"]
    beta_schedule: list[float] = opt_params["betas"]
    max_eval_per_stage: int = opt_params["max_eval"]
    method = opt_params["method"]
    x_init: np.ndarray = opt_params["x_init"]

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

        obj_values.append(final_obj)
        x_values.append(x_init)

        if print_results:
            print(f"Stage {stage_idx} finished. obj = {(final_obj):.4f}")

    # convert to numpy arrays
    obj_values = np.array(obj_values)
    x_values = np.array(x_values)

    assert base_obj_func(x_init, np.zeros(n)) == final_obj, f"base_obj_func(x_init, np.zeros(n)) {base_obj_func(x_init, np.zeros(n))} and final_obj {final_obj} are not equal"
    # assert final_obj == obj_values[-1], f"final_obj {final_obj} and obj_values[-1] {obj_values[-1]} are not equal"
    # assert x_init == x_values[-1], f"x_init {x_init} and x_values[-1] {x_values[-1]} are not equal"

    return x_init, final_obj, obj_values, x_values


def threshold_opt_iterative_wavelength(
    design_dict: dict,
    print_results: bool
    ) -> tuple[np.ndarray, list[float], list[np.ndarray]]:
    design_dicts = get_iterative_wavelength_design_dicts(design_dict)

    n = len(design_dicts)
    
    for i in range(n):
        if i == 0:
            opt_x, final_obj, obj_values, x_values = threshold_opt(
                design_dict=design_dicts[i],
                print_results=print_results
            )
        else:
            design_dict_i = copy.deepcopy(design_dicts[i])
            design_dict_i["opt_params"]["x_init"] = opt_x
            opt_x, final_obj, obj_values, x_values = threshold_opt(
                design_dict=design_dict_i,
                print_results=print_results
            )

    # return the final opt_x, obj_values, x_values
    return opt_x, final_obj, obj_values, x_values


def x_I_opt(
        design_dict: dict, 
        ) -> tuple[np.ndarray, np.ndarray, float]:
    sim_params = design_dict["sim_params"]
    elem_params = design_dict["elem_params"]
    opt_params = design_dict["opt_params"]
    args = design_dict["args"]

    fwd_model = opt_params["forward_model"]

    opt_func = opt_params["opt_func"]

    opt_x, final_obj, obj_values, x_values = opt_func(
        design_dict,
        False
    )

    # Apply the same preprocessing steps that were used during optimization
    opt_x_tensor = torch.tensor(opt_x, dtype=sim_params.dtype, device=sim_params.device)
    
    # Apply density filtering (same as in optimization)
    opt_x_filtered = density_filtering(opt_x_tensor, opt_params["filter_radius"], sim_params)
    
    # Apply hard thresholding with beta = inf (same as final stage of optimization)
    opt_x_proj = heaviside_projection(opt_x_filtered, beta = np.inf, eta = 0.5)
    
    init_params = SimParams(
        Ny=1, 
        Nx=opt_params["x_init"].shape[0]*2, 
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
    
    # Verify that the recomputed objective function matches final_obj
    # Use the same preprocessing and forward model as in optimization
    recomputed_obj = fwd_model(opt_x_proj, sim_params, opt_params, *args)
            
    opt_x = opt_x_proj.detach().cpu().numpy()
    opt_x_full = opt_x_full.detach().cpu().numpy()
            
    return opt_x, opt_x_full, I_opt, final_obj, obj_values, x_values