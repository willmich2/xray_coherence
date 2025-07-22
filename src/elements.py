import torch # type: ignore
import numpy as np # type: ignore
from dataclasses import dataclass 
from src.simparams import SimParams
from src.util import refractive_index_at_wvl
torch.pi = torch.acos(torch.zeros(1)).item() * 2

@dataclass
class ArbitraryElement:
    name: str
    thickness: float
    elem_map: list[np.ndarray]
    gap_map: list[np.ndarray]
    x: torch.Tensor

    def __str__(self):
        return f"ArbitraryElement(name={self.name}, thickness={self.thickness}, elem_map={self.elem_map}, gap_map={self.gap_map})"

    def __copy__(self):
        return ArbitraryElement(
            name=self.name, 
            thickness=self.thickness, 
            elem_map=self.elem_map, 
            gap_map=self.gap_map, 
            x=self.x
            )

    def transmission(self, lam: float, sim_params: SimParams):
        x_tensor = self.x.to(sim_params.device)
        n_elem = refractive_index_at_wvl(lam, self.elem_map)
        n_gap = refractive_index_at_wvl(lam, self.gap_map)
        n_eff = n_elem * x_tensor + n_gap * (1 - x_tensor)
        # Use torch.pi with the correct device
        k0 = 2 * torch.acos(torch.tensor(-1.0, dtype=torch.float32, device=sim_params.device)) / lam 
        
        return torch.exp(1j * k0 * (n_eff - 1) * self.thickness)

@dataclass
class ZonePlate:
    name: str
    thickness: float
    min_feature_size: float
    elem_map: list[np.ndarray]
    gap_map: list[np.ndarray]
    f: float

    def __str__(self):
        return f"ZonePlate(name={self.name}, thickness={self.thickness}, min_feature_size={self.min_feature_size}, elem_map={self.elem_map}, gap_map={self.gap_map})"

    def __copy__(self):
        return ZonePlate(
            name=self.name, 
            thickness=self.thickness, 
            min_feature_size=self.min_feature_size,
            elem_map=self.elem_map,
            gap_map=self.gap_map,
            f=self.f
            )
    
    def transmission(self, lam: float, sim_params: SimParams):
        """
        Calculates the transmission function of the zone plate.
        
        The zone plate pattern is generated up to a maximum radius determined by
        the minimum feature size. Beyond this radius, the transmission is
        that of the gap material.
        """
        pi = torch.acos(torch.tensor(-1.0, dtype=torch.float32, device=sim_params.device))
        n_elem = refractive_index_at_wvl(lam, self.elem_map)
        n_gap = refractive_index_at_wvl(lam, self.gap_map)
        
        # Calculate radial distance from center for all points in the grid
        R_squared = sim_params.X**2 + sim_params.Y**2
        R = torch.sqrt(R_squared)

        R_cutoff = (lam * self.f) / (2 * self.min_feature_size)
        
        # Calculate the path difference to determine the zone number for each point
        path_diff = torch.sqrt(R_squared + self.f**2) - self.f
        zone_number = torch.floor(path_diff / (lam / 2.0))

        # Define the complex transmission for the element and gap materials
        trans_elem = torch.exp(1j * 2 * pi * (n_elem - 1) * self.thickness / lam)
        trans_gap = torch.exp(1j * 2 * pi * (n_gap - 1) * self.thickness / lam)

        # Create the ideal, infinite zone plate pattern based on the zone number
        # Even zones get the gap transmission, odd zones get the element transmission.
        zp_pattern = torch.where(zone_number % 2 == 0, 
                                  trans_gap,
                                  trans_elem)
        
        transmission = torch.where(R <= R_cutoff,
                                   zp_pattern,
                                   trans_gap)
        # --- MODIFICATION END ---
        
        return transmission

@dataclass 
class RectangularElement:
    name: str
    thickness: float
    n_elem: complex
    n_gap: complex
    length: float # in x direction
    width: float # in y direction

    def __str__(self):
        return f"RectangularElement(name={self.name}, thickness={self.thickness}, n_elem={self.n_elem}, n_gap={self.n_gap})"

    def transmission(self, lam: float, n_elem: complex, n_gap: complex, params: SimParams):
        # create tensor of thicknsesses
        thickness_tensor = torch.ones(params.Nx, params.Ny) * self.thickness
        # make thickness tensor 0 for all points outside the element
        thickness_tensor = torch.where(torch.abs(params.X) > self.length/2, 0, thickness_tensor)
        thickness_tensor = torch.where(torch.abs(params.Y) > self.width/2, 0, thickness_tensor)
        n_eff = n_elem * thickness_tensor
        # Use torch.pi with the correct device
        k0 = 2 * torch.acos(torch.tensor(-1.0, dtype=torch.float32, device=params.device)) / lam 
        
        return torch.exp(1j * k0 * n_eff)