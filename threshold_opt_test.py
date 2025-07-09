import numpy as np # type: ignore
import torch # type: ignore
from src.simparams import SimParams
from src.threshold_opt import threshold_opt
from src.forwardmodels import forward_model_focus_plane_wave

print(f"cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Nwvl = 25
lams = [float(x) for x in torch.linspace(400e-9, 500e-9, Nwvl, dtype=torch.float64, device=device)]
weights = [1 / Nwvl]*Nwvl

n_gap = 1.0 + 0j
n_bar = 0.998 + 0.0005j
thickness = 1e-3
z = 0.14e-3

sim_params = SimParams(
    Ny=1, 
    Nx=1600, 
    dx=250e-9, 
    device=device, 
    lams=lams, 
    weights=weights
    )

elem_params = {
    "thickness": thickness, 
    "n_elem": n_bar, 
    "n_gap": n_gap
}

forward_model_args = {
    "Ncenter": 100, 
    "Navg": 150
}

opt_x = threshold_opt(
    sim_params=sim_params, 
    forward_model=forward_model_focus_plane_wave, 
    forward_model_args=(elem_params, forward_model_args, z), 
    beta_schedule=[1.0, 3.0, 5.0, 10.0, 20.0, 25.0, 35.0, 50.0, 100.0, 200.0, 500.0, 1000.0], 
    max_eval_per_stage=500, 
    x_init=np.random.uniform(size=sim_params.Nx)
    )