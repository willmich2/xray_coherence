import torch # type: ignore
import numpy as np # type: ignore
from dataclasses import dataclass 
from src.simparams import SimParams
torch.pi = torch.acos(torch.zeros(1)).item() * 2

@dataclass
class ArbitraryElement:
    name: str
    thickness: float
    n_elem: complex
    n_gap: complex
    x: torch.Tensor

    def __str__(self):
        return f"ArbitraryElement(name={self.name}, thickness={self.thickness}, n_elem={self.n_elem}, n_gap={self.n_gap})"

    def transmission(self, lam: float, n_elem: complex, n_gap: complex, params: SimParams):
        x_tensor = self.x.to(params.device)
        n_eff = n_elem * x_tensor + n_gap * (1 - x_tensor)
        # Use torch.pi with the correct device
        k0 = 2 * torch.acos(torch.tensor(-1.0, dtype=torch.float32, device=params.device)) / lam 
        
        return torch.exp(1j * k0 * n_eff * self.thickness)

@dataclass
class ZonePlate:
    name: str
    thickness: float
    n_elem: complex
    n_gap: complex
    f: float

    def __str__(self):
        return f"ZonePlate(name={self.name}, thickness={self.thickness}, n_elem={self.n_elem}, n_gap={self.n_gap})"
    
    def transmission(self, lam: float, n_elem: complex, n_gap: complex, params: SimParams):
        pi = torch.acos(torch.tensor(-1.0, dtype=torch.float32, device=params.device))
        
        # Calculate radial distance from center
        R = torch.sqrt(params.X**2 + params.Y**2)
        
        # Fresnel zone plate pattern: alternating transparent and opaque zones
        # Zone boundaries occur at r_n = sqrt(n * lam * f + (n * lam * f)^2 / (4 * f^2))
        # For small angles, this simplifies to r_n ≈ sqrt(n * lam * f)
        
        path_diff = torch.sqrt(R**2 + self.f**2) - self.f
        # Calculate zone number for each point
        zone_number = torch.floor(path_diff / (lam / 2.0))
        
        # Create alternating pattern: even zones are transparent (1), odd zones are opaque (0)
        # For phase zone plate: even zones have phase 0, odd zones have phase pi
        transmission = torch.where(zone_number % 2 == 0, 
                                  torch.exp(1j * 2*pi * n_elem * self.thickness / lam),
                                  torch.exp(1j * 2*pi * n_gap * self.thickness / lam))
        
        return transmission

