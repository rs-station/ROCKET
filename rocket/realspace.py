"""Real-space refinement by comparing model-generated maps with target CCP4 maps."""

from collections.abc import Sequence

import gemmi
import numpy as np
import torch
from geomloss import SamplesLoss
from SFC_Torch import PDBParser
from SFC_Torch.mask import reciprocal_grid
from torch import Tensor

from rocket.utils import get_b_from_CC, interpolate_grid_points

from .realspace_utils import create_spherical_mask, get_rscc_from_twomaps


class RealSpace:
    """Calculate loss between target CCP4 map and model-generated map.

    Compares model-generated maps from coordinates with target maps.
    """

    def __init__(
        self,
        target_map,
        pdb_obj: PDBParser,
        device: torch.device,
        mask_position=None,
        mask_radius=None,
        loss_type: str = "cc",
        penalty_center=None,
        penalty_radius=None,
        penalty_weight=100.0,
        save_penalty_mask=False,  # noqa: FBT002
        penalty_mask_path=None,
    ):
        """Initialize RealSpace loss calculator.

        Args:
            target_map: Target CCP4 map
            pdb_obj: PDB parser object
            device: PyTorch device
            mask_position: Center position for spherical mask
            mask_radius: Radius for spherical mask
            loss_type: Type of loss ('cc', 'l2', 'wasserstein',
                'sinkhorn', 'density_explained')
            penalty_center: Center of penalty sphere
            penalty_radius: Radius of penalty sphere
            penalty_weight: Weight for spatial penalty
            save_penalty_mask: Whether to save penalty mask
            penalty_mask_path: Path to save penalty mask

        """
        if loss_type not in [
            "cc",
            "l2",
            "wasserstein",
            "sinkhorn",
            "density_explained",
        ]:
            raise ValueError(
                "loss_type must be 'cc', 'l2', 'wasserstein', "
                "'sinkhorn', or 'density_explained'"
            )

        self.device = device
        self.pdb_obj = pdb_obj
        self.loss_type = loss_type
        self.target_ccp4_map = target_map
        self.mask_position = mask_position
        self.mask_radius = mask_radius

        # Penalty parameters
        self.penalty_center = penalty_center
        self.penalty_radius = penalty_radius
        self.penalty_weight = penalty_weight
        self.save_penalty_mask = save_penalty_mask
        self.penalty_mask_path = penalty_mask_path
        self._penalty_mask_saved = False

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
        """Generate normalized map from model coordinates.

        Args:
            model_coordinates: Model atomic coordinates
            sfc: Structure factor calculator

        Returns:
            Normalized map grid

        """
        Fprotein = sfc.calc_fprotein(model_coordinates, Return=True)
        rs_grid = reciprocal_grid(sfc.Hasu_array, Fprotein, sfc.gridsize)
        map_grid = torch.real(torch.fft.fftn(rs_grid, dim=(-3, -2, -1)))
        map_grid_norm = (map_grid - map_grid.mean()) / map_grid.std()

        if self.is_masked:
            map_grid_norm = map_grid_norm * self.target_mask.float()

        return map_grid_norm

    def get_correlation(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """Calculate correlation coefficient between target and model maps.

        Args:
            model_coordinates: Model atomic coordinates
            dcp: Structure factor calculator

        Returns:
            Correlation coefficient

        """
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

    def forward(
        self,
        model_coordinates: torch.Tensor,
        dcp,
        NO_RBR=True,  # noqa: FBT002, N803
    ) -> torch.Tensor:
        """Compute loss based on selected loss type.

        Args:
            model_coordinates: Model atomic coordinates
            dcp: Structure factor calculator
            NO_RBR: Whether rigid body refinement is disabled

        Returns:
            Loss value (and optionally B-factors)

        """
        if self.loss_type == "cc":
            return self._forward_cc(model_coordinates, dcp)
        elif self.loss_type == "l2":
            return self._forward_l2(
                model_coordinates,
                dcp,
                NO_RBR=NO_RBR,
                penalty_center=self.penalty_center,
                penalty_radius=self.penalty_radius,
                penalty_weight=self.penalty_weight,
                save_penalty_mask=self.save_penalty_mask
                and not self._penalty_mask_saved,
                penalty_mask_path=self.penalty_mask_path,
            )
        elif self.loss_type == "wasserstein":
            return self._forward_wasserstein(model_coordinates, dcp)
        elif self.loss_type == "sinkhorn":
            return self._forward_sinkhorn(
                model_coordinates, dcp, voxel_size=self.pixel_size
            )
        elif self.loss_type == "density_explained":
            return self._forward_density_explained(model_coordinates, dcp)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _forward_cc(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """CC loss (negative correlation coefficient)."""
        return -self.get_correlation(model_coordinates, dcp)

    def _forward_l2(
        self,
        model_coordinates: torch.Tensor,
        dcp,
        NO_RBR=True,  # noqa: FBT002, N803
        penalty_center=None,
        penalty_radius=None,
        penalty_weight=100.0,
        save_penalty_mask=False,  # noqa: FBT002
        penalty_mask_path=None,
    ) -> torch.Tensor:
        """L2 loss on normalized maps with optional spatial penalty."""
        model_map = self.model2map(model_coordinates, dcp)

        if model_map.shape != self.target_map_grid.shape:
            raise ValueError(
                f"Grid shape mismatch: {model_map.shape} vs "
                f"{self.target_map_grid.shape}"
            )

        # Extract masked regions FIRST
        target_masked = self.target_map_grid[self.target_mask]
        model_masked = model_map[self.target_mask]

        if target_masked.numel() < 50:
            if NO_RBR:
                return (
                    torch.tensor(1e6, device=self.device, dtype=torch.float32),
                    None,
                )
            return (
                torch.tensor(1e6, device=self.device, dtype=torch.float32),
                None,
            )

        # Compute normalization statistics ONLY from masked region
        target_mean, target_std = target_masked.mean(), target_masked.std()
        model_mean, model_std = model_masked.mean(), model_masked.std()

        # Normalize the FULL 3D maps using masked region statistics
        target_normalized_3d = (self.target_map_grid - target_mean) / (
            target_std + 1e-8
        )
        model_normalized_3d = (model_map - model_mean) / (model_std + 1e-8)

        # Extract masked region from normalized maps for loss computation
        target_normalized = target_normalized_3d[self.target_mask]
        model_normalized = model_normalized_3d[self.target_mask]

        # Compute base L2 loss on masked region
        l2_loss = torch.mean((target_normalized - model_normalized) ** 2)

        # Add spatial penalty if specified
        if penalty_center is not None and penalty_radius is not None:
            spatial_penalty = self._compute_spatial_penalty(
                model_coordinates,
                penalty_center,
                penalty_radius,
                penalty_weight,
                save_mask=save_penalty_mask,
                mask_path=penalty_mask_path,
            )
            total_loss = l2_loss + spatial_penalty
            print(
                f"L2 loss: {l2_loss:.4f}, "
                f"Spatial penalty: {spatial_penalty:.4f}, "
                f"Total: {total_loss:.4f}"
            )
        else:
            total_loss = l2_loss

        # RSCC map in 3D (only if NO_RBR)
        if NO_RBR:
            # Smooth FULL 3D maps with 2 Angstrom Gaussian before
            # computing correlation
            target_smoothed = self._gaussian_smooth_3d(
                target_normalized_3d, sigma_angstrom=4.0
            )
            model_smoothed = self._gaussian_smooth_3d(
                model_normalized_3d, sigma_angstrom=4.0
            )

            # RSCC map in 3D using smoothed maps
            ccmap = get_rscc_from_twomaps(target_smoothed, model_smoothed.detach())

            # Interpolate atom positions on 3D ccmap
            atom_cc = interpolate_grid_points(
                ccmap, dcp.atom_pos_frac.detach().cpu().numpy()
            )
            atom_cc = np.clip(atom_cc, -0.99, 0.99)

            rscc_bfactors = torch.tensor(
                get_b_from_CC(atom_cc, dcp.dmin),
                dtype=torch.float32,
                device=self.device,
            )

            return total_loss, rscc_bfactors
        else:
            return total_loss, None

    def _make_coords(self, shape, voxel_size, device=None, dtype=None) -> Tensor:
        """Create coordinates for each voxel center in R^3.

        Returns coordinates [N, 3] centered so grid's geometric center is at
        origin.

        Args:
            shape: Grid dimensions (Dz, Dy, Dx)
            voxel_size: Voxel size in each dimension (vz, vy, vx)
            device: PyTorch device
            dtype: PyTorch dtype

        Returns:
            Coordinate array [N, 3]

        """
        Dz, Dy, Dx = shape
        vz, vy, vx = voxel_size
        z = (torch.arange(Dz, device=device, dtype=dtype) - (Dz - 1) / 2) * vz
        y = (torch.arange(Dy, device=device, dtype=dtype) - (Dy - 1) / 2) * vy
        x = (torch.arange(Dx, device=device, dtype=dtype) - (Dx - 1) / 2) * vx
        Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        return coords

    def _forward_sinkhorn(
        self,
        model_coordinates: torch.Tensor,
        dcp,
        *,
        voxel_size: Sequence[float],
        blurs: Sequence[float] = (3.0, 2.0, 1.0, 0.5),
        p: int = 2,
        debias: bool = True,
        backend: str = "multiscale",
        max_points_tensorized: int = 20000,  # noqa: ARG002
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """EM Sinkhorn loss.

        Args:
            model_coordinates: Model atomic coordinates
            dcp: Structure factor calculator
            voxel_size: Voxel size (dz, dy, dx)
            blurs: Multiscale schedule (Å if voxel_size is in Å)
            p: Cost power (2 = quadratic)
            debias: Use Sinkhorn divergence
            backend: Backend ('multiscale', 'online', or 'tensorized')
            max_points_tensorized: Subsample limit for tensorized backend
            eps: Numerical stability epsilon

        Returns:
            Sinkhorn loss value

        """
        # Build model map
        model_map = self.model2map(model_coordinates, dcp)
        if model_map.shape != self.target_map_grid.shape:
            raise ValueError(
                f"Grid shape mismatch: {model_map.shape} \n"
                f"vs {self.target_map_grid.shape}"
            )

        # Coordinates (N,3) in the right units
        coords = self._make_coords(
            self.target_map_grid.shape,
            voxel_size=voxel_size,
            device=self.device,
            dtype=torch.float32,
        )

        # Nonnegative masked masses, normalized
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
                loss="sinkhorn",
                p=p,
                blur=_blur,
                debias=debias,
                backend=backend,
                scaling=0.9,
            )
            return sl(a_, x_, b_, x_)

        # Multiscale: average over blur schedule
        loss_val = 0.0
        for blur in blurs:
            loss_val = loss_val + _sinkhorn_one_blur(float(blur), x, a, b)
        return loss_val / float(len(blurs))

    def _forward_density_explained(
        self, model_coordinates: torch.Tensor, dcp
    ) -> torch.Tensor:
        """Loss that maximizes density explained by the model.

        Creates a 1.5 Å mask around each atom and compares target vs model
        density. Returns negative of explained density (minimizing =
        maximizing explained density).

        Args:
            model_coordinates: Model atomic coordinates
            dcp: Structure factor calculator

        Returns:
            Negative of explained density

        """
        model_map = self.model2map(model_coordinates, dcp)

        if model_map.shape != self.target_map_grid.shape:
            raise ValueError(
                f"Grid shape mismatch: {model_map.shape} vs "
                f"{self.target_map_grid.shape}"
            )

        # Create mask around atoms (1.5 Å radius)
        atom_mask = self._create_atomic_mask(model_coordinates, radius=1.5)

        # Extract densities within atomic mask
        target_density = self.target_map_grid[atom_mask]
        model_density = model_map[atom_mask]

        if target_density.numel() < 10:
            return torch.tensor(1e6, device=self.device, dtype=torch.float32)

        # Compute explained density as correlation within mask
        correlation = torch.corrcoef(torch.stack([target_density, model_density]))[0, 1]

        # Also add a term for total density captured
        total_target = target_density.sum()
        total_model = model_density.sum()

        # Penalize if model doesn't capture enough density
        density_ratio = total_model / (total_target + 1e-8)

        # Combined loss: negative correlation + penalty for missing density
        loss = -correlation + 0.5 * torch.abs(1.0 - density_ratio)

        return loss

    def _create_atomic_mask(
        self, model_coordinates: torch.Tensor, radius: float = 1.5
    ) -> torch.Tensor:
        """Create a boolean mask of voxels within radius (Å) of any atom.

        Args:
            model_coordinates: Atomic coordinates [N_atoms, 3] in orthogonal
                space
            radius: Mask radius in Angstroms

        Returns:
            Boolean mask [Dz, Dy, Dx]

        """
        # Get grid shape and voxel size
        grid_shape = self.target_map_grid.shape
        voxel_size = torch.tensor(self.pixel_size, device=self.device)

        # Create coordinate grid for all voxels
        nx, ny, nz = grid_shape
        x_coords = torch.arange(nx, device=self.device, dtype=torch.float32)
        y_coords = torch.arange(ny, device=self.device, dtype=torch.float32)
        z_coords = torch.arange(nz, device=self.device, dtype=torch.float32)

        xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

        # Convert voxel indices to orthogonal coordinates
        grid_coords_orth = torch.stack(
            [
                xx * voxel_size[0],
                yy * voxel_size[1],
                zz * voxel_size[2],
            ],
            dim=-1,
        )

        # Get unit cell origin from target map
        origin = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        grid_coords_orth = grid_coords_orth + origin

        # Initialize mask
        mask = torch.zeros(grid_shape, dtype=torch.bool, device=self.device)

        # For each atom, mark voxels within radius
        for atom_pos in model_coordinates:
            distances = torch.norm(grid_coords_orth - atom_pos, dim=-1)
            mask = mask | (distances <= radius)

        return mask

    @property
    def is_masked(self):
        """Return True if masking is applied."""
        return self.mask_position is not None and self.mask_radius is not None

    def _gaussian_smooth_3d(
        self, map_3d: torch.Tensor, sigma_angstrom: float = 2.0
    ) -> torch.Tensor:
        """Apply Gaussian smoothing using FFT (fast for large grids).

        Args:
            map_3d: 3D tensor [Dz, Dy, Dx]
            sigma_angstrom: Standard deviation of Gaussian kernel in
                Angstroms

        Returns:
            Smoothed 3D map

        """
        # Convert sigma from Angstroms to voxels
        voxel_size = torch.tensor(self.pixel_size, device=self.device)
        sigma_voxels = sigma_angstrom / voxel_size

        # Create Gaussian kernel in Fourier space
        nz, ny, nx = map_3d.shape

        # Frequency grids
        kz = torch.fft.fftfreq(nz, d=1.0, device=self.device)
        ky = torch.fft.fftfreq(ny, d=1.0, device=self.device)
        kx = torch.fft.fftfreq(nx, d=1.0, device=self.device)

        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")

        # Gaussian in Fourier space: exp(-2π²σ²k²)
        K2 = (
            (KZ / sigma_voxels[0]) ** 2
            + (KY / sigma_voxels[1]) ** 2
            + (KX / sigma_voxels[2]) ** 2
        )
        gaussian_filter = torch.exp(-2 * (torch.pi**2) * K2)

        # Apply filter in Fourier space
        map_fft = torch.fft.fftn(map_3d)
        smoothed_fft = map_fft * gaussian_filter
        smoothed = torch.fft.ifftn(smoothed_fft).real

        return smoothed

    def _compute_spatial_penalty(
        self,
        model_coordinates: torch.Tensor,
        penalty_center: torch.Tensor,
        penalty_radius: float,
        penalty_weight: float = 100.0,
        save_mask: bool = False,  # noqa: FBT002
        mask_path: str = None,
    ) -> torch.Tensor:
        """Compute penalty for atoms within a spherical region.

        Args:
            model_coordinates: Atomic coordinates [N_atoms, 3]
            penalty_center: Center of penalty sphere [3]
            penalty_radius: Radius of penalty sphere (Angstroms)
            penalty_weight: Scaling factor for penalty
            save_mask: Whether to save the penalty mask as CCP4
            mask_path: Path to save the mask

        Returns:
            Penalty value

        """
        # Ensure penalty_center is on the correct device
        if not isinstance(penalty_center, torch.Tensor):
            penalty_center = torch.tensor(
                penalty_center, device=self.device, dtype=torch.float32
            )
        else:
            penalty_center = penalty_center.to(self.device)

        # Compute distances from all atoms to penalty center
        distances = torch.norm(model_coordinates - penalty_center, dim=-1)

        # Save 3D penalty mask if requested
        if save_mask:
            self._save_penalty_mask_ccp4(penalty_center, penalty_radius, mask_path)

        # Soft penalty using sigmoid (smooth gradients)
        softness = 1.0
        penalty = penalty_weight * torch.sum(
            torch.sigmoid((penalty_radius - distances) / softness)
        )

        return penalty

    def _save_penalty_mask_ccp4(
        self,
        penalty_center: torch.Tensor,
        penalty_radius: float,
        mask_path: str = None,
    ):
        """Save penalty mask as a CCP4 file.

        Args:
            penalty_center: Center of penalty sphere [3]
            penalty_radius: Radius of penalty sphere (Angstroms)
            mask_path: Output path (default: penalty_mask.ccp4)

        """
        if mask_path is None:
            mask_path = "penalty_mask.ccp4"

        # Get grid shape and create coordinate grid
        grid_shape = self.target_map_grid.shape
        nx, ny, nz = grid_shape

        # Create voxel coordinate grid
        voxel_size = torch.tensor(self.pixel_size, device=self.device)
        x_coords = torch.arange(nx, device=self.device, dtype=torch.float32)
        y_coords = torch.arange(ny, device=self.device, dtype=torch.float32)
        z_coords = torch.arange(nz, device=self.device, dtype=torch.float32)

        xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

        # Convert to orthogonal coordinates
        grid_coords_orth = torch.stack(
            [
                xx * voxel_size[0],
                yy * voxel_size[1],
                zz * voxel_size[2],
            ],
            dim=-1,
        )

        # Compute distance from penalty center for each voxel
        distances = torch.norm(grid_coords_orth - penalty_center, dim=-1)

        # Create mask: 1.0 inside sphere, 0.0 outside
        mask_grid = (distances <= penalty_radius).float()

        # Convert to numpy
        mask_np = mask_grid.cpu().numpy().astype(np.float32)

        # Create gemmi grid matching target map
        unit_cell = self.target_ccp4_map.grid.unit_cell
        spacegroup = self.target_ccp4_map.grid.spacegroup

        # Create new grid
        grid = gemmi.FloatGrid(nx, ny, nz)
        grid.set_unit_cell(unit_cell)
        grid.spacegroup = spacegroup

        # Copy mask data
        np.copyto(np.array(grid, copy=False), mask_np)

        # Create CCP4 map
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        ccp4.update_ccp4_header()

        # Write to file
        ccp4.write_ccp4_map(mask_path)

        print(f"Saved penalty mask to {mask_path}")
        print(f"  - Penalty center: {penalty_center.cpu().numpy()}")
        print(f"  - Penalty radius: {penalty_radius} Å")
        print(f"  - Grid shape: {grid_shape}")
        print(f"  - Voxels inside: {mask_grid.sum().item():.0f}")
