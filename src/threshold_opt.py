import numpy as np # type: ignore
import torch # type: ignore
import nlopt # type: ignore
from typing import Callable
from src.inversedesign_utils import create_objective_function, heaviside_projection
from src.simparams import SimParams
from src.forwardmodels import field_z_arbg_z, field_z_arbg_z_mode, forward_model_focus_plane_wave_power, forward_model_focus_incoherent_mc_power, propagate_z1_arbg_z2, forward_model_focus_incoherent_gaussian_schell_power
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
    ) -> np.ndarray:

    # compiled_model = torch.compile(forward_model, mode="default")

    for stage_idx, beta_val in enumerate(beta_schedule, 1):
        if print_results:
            print(f"\n--- Stage {stage_idx} with beta = {beta_val} ---")

        # Create NLopt optimizer
        n = x_init.shape[0]
        opt = nlopt.opt(method, n)
        
        # Set objective function for this stage, using the current beta
        opt.set_max_objective(create_objective_function(
            beta=beta_val, 
            forward_model=forward_model, 
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

        opt.set_param("inner_maxeval", opt_params["inner_maxeval"])

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
    
    return x_init


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
    
    opt_x = threshold_opt(
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

    if fwd_model == forward_model_focus_plane_wave_power:
    
        U_opt = field_z_arbg_z(
            x = opt_x_full, 
            sim_params = sim_params, 
            elem_params = elem_params, 
            z = args[-1]
            )

        weights_t = sim_params.weights.view(-1, 1, 1)
        I_opt = torch.sum((U_opt.abs()**2) * weights_t, dim=0).reshape(sim_params.Nx).detach().cpu().numpy()
    elif fwd_model == forward_model_focus_incoherent_mc_power:
        element = ArbitraryElement(
            name="ArbitraryElement", 
            thickness=elem_params["thickness"], 
            elem_map=elem_params["elem_map"], 
            gap_map=elem_params["gap_map"], 
            x=opt_x_full
        )

        I_mc = mc_propagate_accumulate_intensity(
            u_init_func=gaussian_source,
            u_init_func_args=(args[4],),
            prop_func=propagate_z1_arbg_z2,
            prop_func_args=(args[3], element),
            n=args[5],
            z=args[2],
            sim_params=sim_params
        )
        weights_t = sim_params.weights.view(-1, 1, 1)
        I_opt = torch.sum(I_mc * weights_t, dim=0).reshape(sim_params.Nx).detach().cpu().numpy()
    elif fwd_model == forward_model_focus_incoherent_gaussian_schell_power:
        element = ArbitraryElement(
            name="ArbitraryElement", 
            thickness=elem_params["thickness"], 
            elem_map=elem_params["elem_map"], 
            gap_map=elem_params["gap_map"], 
            x=opt_x_full
        )
        I_opt = gaussian_schell_propagate_accumulate_intensity(
            sim_params=sim_params,
            lam_n=args[4],
            psi_n=args[5],
            z1=args[2],
            z2=args[3],
            element=element
        )
    else:
        input_modes = args[-2]
        input_eigen_vals = args[-1]
        Nmodes = input_modes.shape[0]
        output_modes = torch.zeros((Nmodes, sim_params.weights.shape[0], sim_params.Ny, sim_params.Nx), dtype=sim_params.dtype, device=sim_params.device)

        for i in range(Nmodes):
            Uzgz = field_z_arbg_z_mode(opt_x_full, input_modes[i], sim_params, elem_params, args[2])
            output_modes[i] = Uzgz

        output_modes = output_modes.transpose(0, 1)
        input_eigen_vals = input_eigen_vals.transpose(0, 1)
        
        I_arr = torch.sum(output_modes.abs().pow(2)*(input_eigen_vals.unsqueeze(-1).unsqueeze(-1)), dim=1)
        I_opt = torch.sum(I_arr * sim_params.weights.unsqueeze(-1).unsqueeze(-1), dim=0).reshape(sim_params.Nx).detach().cpu().numpy()
        del output_modes
        del I_arr
        
    opt_x = opt_x_proj.detach().cpu().numpy()
    opt_x_full = opt_x_full.detach().cpu().numpy()
            
    return opt_x, opt_x_full, I_opt, final_obj