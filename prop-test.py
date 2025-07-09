import numpy as np # type: ignore
import torch # type: ignore
from simparams import SimParams
from propagation import propagate_z
from sources import gaussian_source

print(f"cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dx = 250e-9 # grid spacing
W = 0.4e-3 # width of the simulation region
Nx = int(W / dx)
Ny = 1
print(f"Nx: {Nx}")
print(f"Ny: {Ny}")
Nwvl = 25
lams = [float(x) for x in torch.linspace(400e-9, 500e-9, Nwvl, dtype=torch.float64, device=device)]
weights = [1 / Nwvl]*Nwvl
rsrc = 1e-6
nmc = 800
n_gap = 1.0 + 0j
n_bar = 0.998 + 0.0005j
thickness = 1e-3
z = 0.14e-3

sim_params = SimParams(Ny, Nx, dx, device, lams, weights, nmc)
source = gaussian_source(rsrc, sim_params)

U_z = propagate_z(source, z, sim_params)

print("propagation complete successfully")

# save U_z as npy file
np.save("/home/gridsan/wmichaels/xray-coherence-sim/U_z.npy", U_z.cpu().numpy())

