"""
Real-space refinement by comparing model-generated maps with target CCP4 maps.
"""

from collections.abc import Sequence

import numpy as np
import torch
from geomloss import SamplesLoss
from SFC_Torch import PDBParser
from SFC_Torch.mask import reciprocal_grid
from torch import Tensor

from .realspace_utils import create_spherical_mask


class RealSpace:
    """
    Calculate loss between target CCP4 map and model-generated map from coordinates.
    """

    def __init__(
        self,
        target_map,
        pdb_obj: PDBParser,
        device: torch.device,
        mask_position=None,
        mask_radius=None,
        loss_type: str = "cc",
    ):
        """Initialize RealSpace loss calculator"""
        if loss_type not in ["cc", "l2", "wasserstein", "sinkhorn"]:
            raise ValueError(
                "loss_type must be 'cc', 'l2', 'wasserstein', or 'sinkhorn'"
            )

        self.device = device
        self.pdb_obj = pdb_obj
        self.loss_type = loss_type
        self.target_ccp4_map = target_map
        self.mask_position = mask_position
        self.mask_radius = mask_radius

        # Load target map grid
        target_grid_np = np.array(target_map.grid, copy=False)
        self.target_map_grid = torch.tensor(
            target_grid_np, device=device, dtype=torch.float32
        ).clone()

        if mask_position is not None and mask_radius is not None:
            mask_array = create_spherical_mask(
                target_map.grid, mask_position, mask_radius
            )
            print("mask array", mask_array.shape)
            self.target_mask = torch.tensor(mask_array, device=device, dtype=torch.bool)
        else:
            self.target_mask = torch.ones_like(self.target_map_grid, dtype=torch.bool)

        # Set all atoms as valid for alignment
        total_atoms = len(pdb_obj.atom_pos)
        self.valid_atom_indices = np.arange(total_atoms)
        self.ind1 = self.valid_atom_indices
        self.ind2 = self.valid_atom_indices
        self.pixel_size = [
            target_map.grid.unit_cell.a / target_map.grid.nu,
            target_map.grid.unit_cell.b / target_map.grid.nv,
            target_map.grid.unit_cell.c / target_map.grid.nw,
        ]

    def model2map(self, model_coordinates: torch.Tensor, sfc) -> torch.Tensor:
        """Generate normalized map from model coordinates"""
        Fprotein = sfc.calc_fprotein(model_coordinates, Return=True)
        rs_grid = reciprocal_grid(sfc.Hasu_array, Fprotein, sfc.gridsize)
        map_grid = torch.real(torch.fft.fftn(rs_grid, dim=(-3, -2, -1)))
        map_grid_norm = (map_grid - map_grid.mean()) / map_grid.std()

        if self.is_masked:
            map_grid_norm = map_grid_norm * self.target_mask.float()

        return map_grid_norm

    def get_correlation(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """Calculate correlation coefficient between target and model maps"""
        model_map = self.model2map(model_coordinates, dcp)

        if model_map.shape != self.target_map_grid.shape:
            raise ValueError(
                f"Grid shape mismatch: {model_map.shape} vs "
                f"{self.target_map_grid.shape}"
            )

        target_masked = self.target_map_grid[self.target_mask]
        model_masked = model_map[self.target_mask]

        correlation = torch.corrcoef(torch.stack([target_masked, model_masked]))[0, 1]
        return correlation

    def forward(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """Compute loss based on selected loss type"""
        if self.loss_type == "cc":
            return self._forward_cc(model_coordinates, dcp)
        elif self.loss_type == "l2":
            return self._forward_l2(model_coordinates, dcp)
        elif self.loss_type == "wasserstein":
            return self._forward_wasserstein(model_coordinates, dcp)
        elif self.loss_type == "sinkhorn":
            return self._forward_sinkhorn(
                model_coordinates, dcp, voxel_size=self.pixel_size
            )

    def _forward_cc(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """CC loss (negative correlation coefficient)"""
        return -self.get_correlation(model_coordinates, dcp)

    def _forward_l2(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """L2 loss on normalized maps"""
        model_map = self.model2map(model_coordinates, dcp)

        if model_map.shape != self.target_map_grid.shape:
            raise ValueError(
                f"Grid shape mismatch: {model_map.shape} vs "
                f"{self.target_map_grid.shape}"
            )

        target_masked = self.target_map_grid[self.target_mask]
        model_masked = model_map[self.target_mask]

        if target_masked.numel() < 50:
            return torch.tensor(1e6, device=self.device, dtype=torch.float32)

        # Normalize both maps using same statistics
        target_mean, target_std = target_masked.mean(), target_masked.std()
        model_mean, model_std = model_masked.mean(), model_masked.std()

        if target_std > 1e-8 and model_std > 1e-8:
            target_normalized = (target_masked - target_mean) / target_std
            model_normalized = (model_masked - model_mean) / model_std
        else:
            target_normalized = target_masked - target_mean
            model_normalized = model_masked - model_mean

        return torch.mean((target_normalized - model_normalized) ** 2)

    def _make_coords(self, shape, voxel_size, device=None, dtype=None) -> Tensor:
        """
        Returns coordinates for each voxel center in R^3, shape [N, 3],
        centered so that the grid's geometric center is at the origin.
        """
        Dz, Dy, Dx = shape
        vz, vy, vx = voxel_size
        z = (torch.arange(Dz, device=device, dtype=dtype) - (Dz - 1) / 2) * vz
        y = (torch.arange(Dy, device=device, dtype=dtype) - (Dy - 1) / 2) * vy
        x = (torch.arange(Dx, device=device, dtype=dtype) - (Dx - 1) / 2) * vx
        Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # [N,3]
        return coords

    def _forward_sinkhorn(
        self,
        model_coordinates: torch.Tensor,
        dcp,
        *,
        # Geometry / units
        voxel_size: Sequence[float],  # (dz, dy, dx).
        # Multiscale schedule (Å if voxel_size is in Å)
        blurs: Sequence[float] = (3.0, 2.0, 1.0, 0.5),
        p: int = 2,  # quadratic cost
        debias: bool = True,  # Use Sinkhorn divergence
        # Backends + safety
        backend: str = "multiscale",  # "multiscale" (KeOps), "online", or "tensorized"
        max_points_tensorized: int = 20000,  # subsample if tensorized to avoid OOM
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        EM Sinkhorn loss:
        - clamp maps to >=0, apply mask, normalize masses
        - multiscale Sinkhorn divergence across several blurs
        - scalable backend (KeOps if available) or safe subsampling for tensorized
        """

        # 1) Build model map
        model_map = self.model2map(model_coordinates, dcp)
        if model_map.shape != self.target_map_grid.shape:
            raise ValueError(
                f"Grid shape mismatch: {model_map.shape} \n"
                f"vs {self.target_map_grid.shape}"
            )

        # 3) Coordinates (N,3) in the right units
        coords = self._make_coords(
            self.target_map_grid.shape,
            voxel_size=voxel_size,
            device=self.device,
            dtype=torch.float32,
        )  # (N,3)

        # 4) Nonnegative masked masses, normalized
        if self.is_masked:
            mask = self.target_mask.to(self.device)
            tgt = torch.clamp(self.target_map_grid, min=0) * mask
            mdl = torch.clamp(model_map, min=0) * mask
            active = mask.reshape(-1) > 0
        else:
            tgt = torch.clamp(self.target_map_grid, min=0)
            mdl = torch.clamp(model_map, min=0)
            active = torch.ones_like(tgt, dtype=torch.bool).reshape(-1)

        a = tgt.reshape(-1)[active].to(dtype=torch.float32)
        b = mdl.reshape(-1)[active].to(dtype=torch.float32)
        x = coords[active].to(dtype=torch.float32)

        # Normalize to probability masses
        a = a / (a.sum() + eps)
        b = b / (b.sum() + eps)

        # Helper to run one blur safely
        def _sinkhorn_one_blur(_blur: float, x_, a_, b_) -> torch.Tensor:
            sl = SamplesLoss(
                loss="sinkhorn",  # divergence via debias=True
                p=p,
                blur=_blur,
                debias=debias,
                backend=backend,  # "multiscale" (KeOps) recommended
                scaling=0.9,
            )

            # GeomLoss 4-arg API is (alpha, x, beta, y). Same support -> pass x twice.
            return sl(a_, x_, b_, x_)

        # 6) Multiscale: average over blur schedule
        loss_val = 0.0
        for blur in blurs:
            loss_val = loss_val + _sinkhorn_one_blur(float(blur), x, a, b)
        return loss_val / float(len(blurs))

    @property
    def is_masked(self):
        """Return True if masking is applied"""
        return self.mask_position is not None and self.mask_radius is not None
