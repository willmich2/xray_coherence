import torch # type: ignore
from dataclasses import dataclass 

@dataclass
class SimParams:
    Ny: int
    Nx: int
    dx: float
    device: torch.device
    dtype: torch.dtype
    lams: list[float]
    weights: list[float]

    def __post_init__(self):
        zero = torch.zeros(0, dtype=self.dtype, device=self.device)
        self.x = torch.linspace(-self.Nx/2, self.Nx/2, steps=self.Nx, dtype=zero.real.dtype, device=self.device) * self.dx
        self.y = torch.linspace(-self.Ny/2, self.Ny/2, steps=self.Ny, dtype=zero.real.dtype, device=self.device) * self.dx
        self.Y, self.X = torch.meshgrid(self.y, self.x, indexing='ij')

    def __str__(self):
        return f"SimParams(Ny={self.Ny}, Nx={self.Nx}, dx={self.dx}, device={self.device}, lams={self.lams}, weights={self.weights}"

    def copy(self):
        return SimParams(
            Ny = self.Ny, 
            Nx = self.Nx, 
            dx = self.dx, 
            device = self.device, 
            dtype = self.dtype, 
            lams = self.lams, 
            weights = self.weights
            )
