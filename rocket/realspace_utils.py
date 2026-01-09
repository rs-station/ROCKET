"""Utility functions for real-space refinement operations."""

import time

import gemmi
import numpy as np
import torch

from rocket.coordinates import quaternions_to_SO3


def get_rscc_from_twomaps(map1: torch.Tensor, map2: torch.Tensor) -> torch.Tensor:
    """Compute voxel-wise real-space correlation coefficient between two maps.

    Returns:
        Correlation map (same shape as input maps)

    """
    # Normalize both maps (zero mean, unit variance)
    map1_normed = (map1 - map1.mean()) / (map1.std() + 1e-8)
    map2_normed = (map2 - map2.mean()) / (map2.std() + 1e-8)

    # The correlation coefficient is simply the product of normalized maps
    # averaged over all voxels: CC = E[Z1 * Z2] where Z1, Z2 are z-scores
    ccmap = map1_normed * map2_normed

    return ccmap.cpu().numpy()


def rigidbody_refine_quat(
    xyz,
    realspaceloss,
    dcp,
    cra_name,
    lbfgs=False,  # noqa: FBT002
    added_chain_HKL=None,
    added_chain_asu=None,
    lbfgs_lr=150.0,
    verbose=True,  # noqa: FBT002
    domain_segs=None,
):
    """Perform rigid body refinement using quaternions."""
    resid = [int(i.split("-")[1]) + 1 for i in cra_name]
    minid = min(resid)
    maxid = max(resid)

    if domain_segs is None:
        domain_ranges = [[minid, maxid + 1]]
    else:
        domain_ranges = []
        start = minid
        for i, seg in enumerate(domain_segs):
            domain_ranges.append([start, seg])
            start = seg
            if i == len(domain_segs) - 1:
                domain_ranges.append([start, maxid + 1])

    domain_bools = []
    for domain_start, domain_end_notin in domain_ranges:
        domain_bools.append(
            np.array([(i >= domain_start) and (i < domain_end_notin) for i in resid])
        )
    n_domains = len(domain_bools)

    # llgloss.sfc.get_scales_lbfgs()
    if lbfgs:
        trans_vecs, qs, loss_track_pose = find_rigidbody_matrix_lbfgs_quat(
            realspaceloss,
            dcp,
            xyz.detach(),
            realspaceloss.device,
            domain_bools,
            added_chain_HKL=added_chain_HKL,
            added_chain_asu=added_chain_asu,
            lbfgs_lr=lbfgs_lr,
            verbose=verbose,
        )
    else:
        pass
        # trans_vec, q, loss_track_pose = find_rigidbody_matrix_adam_quat(
        #     llgloss,
        #     propose_coms.clone().detach(),
        #     propose_rmcom.clone().detach(),
        #     llgloss.device,
        #     added_chain_HKL=added_chain_HKL,
        #     added_chain_asu=added_chain_asu,
        #     verbose=verbose
        # )
    optimized_xyz = torch.ones_like(xyz)
    for i in range(n_domains):
        propose_rmcom = xyz[domain_bools[i]] - torch.mean(xyz[domain_bools[i]], dim=0)
        propose_com = torch.mean(xyz[domain_bools[i]], dim=0)
        transform_i = quaternions_to_SO3(qs[i]).detach()
        optimized_xyz[domain_bools[i]] = (
            torch.matmul(propose_rmcom, transform_i)
            + propose_com
            + trans_vecs[i].detach()
        )

    return optimized_xyz, loss_track_pose


def find_rigidbody_matrix_lbfgs_quat(
    realspaceloss,
    dcp,
    xyz,
    device,
    domain_bools,
    added_chain_HKL=None,
    added_chain_asu=None,
    lbfgs_lr=150.0,
    verbose=True,  # noqa: FBT002
):
    """Find rigid body transformation matrix using LBFGS optimization."""
    n_domains = len(domain_bools)
    qs = [
        torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True
        )
        for _ in range(n_domains)
    ]
    trans_vecs = [
        torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
        for _ in range(n_domains)
    ]

    loss_track_pose = pose_train_lbfgs_quat(
        realspaceloss,
        dcp,
        qs,
        trans_vecs,
        xyz,
        domain_bools,
        loss_track=[],
        lr=lbfgs_lr,
        added_chain_HKL=added_chain_HKL,
        added_chain_asu=added_chain_asu,
        verbose=verbose,
    )
    return trans_vecs, qs, loss_track_pose


def pose_train_lbfgs_quat(
    realspaceloss,
    dcp,
    qs,
    trans_vecs,
    xyz,
    domain_bools,
    lr=150.0,
    n_steps=15,
    loss_track=None,
    added_chain_HKL=None,
    added_chain_asu=None,
    verbose=True,  # noqa: FBT002, ARG001
):
    """Train pose using LBFGS optimizer with quaternions."""
    if loss_track is None:
        loss_track = []

    def closure():
        optimizer.zero_grad()
        temp_model = torch.zeros_like(xyz)
        for i in range(n_domains):
            temp_R = quaternions_to_SO3(qs[i])
            temp_model[domain_bools[i]] = (
                torch.matmul(propose_rmcoms[i], temp_R)
                + propose_coms[i]
                + trans_vecs[i]
            )
        result = realspaceloss.forward(temp_model, dcp, NO_RBR=False)
        loss = result[0] if isinstance(result, tuple) else result
        loss.backward()
        return loss

    n_domains = len(domain_bools)
    optimizer = torch.optim.LBFGS(
        qs + trans_vecs,
        lr=lr,
        line_search_fn="strong_wolfe",
        tolerance_change=1e-9,
        max_iter=1,
    )
    propose_rmcoms = []
    propose_coms = []
    for domain_bool in domain_bools:
        propose_rmcoms.append(xyz[domain_bool] - torch.mean(xyz[domain_bool], dim=0))
        propose_coms.append(torch.mean(xyz[domain_bool], dim=0))
    start_time = time.time()
    for _ in range(n_steps):
        temp = optimizer.step(closure)
        loss_track.append(temp.item())
    elapsed_time = time.time() - start_time
    if verbose:
        print(
            f"LBFGS RBR, {n_steps} steps, time taken: {elapsed_time:.4f} seconds",
            flush=True,
        )
    return loss_track


def create_spherical_mask(
    map_grid: gemmi.FloatGrid, position: np.ndarray, radius: float
) -> np.ndarray:
    """Create spherical boolean mask for a map grid.

    Args:
        map_grid: Input map grid (used for dimensions and metadata)
        position: Center position for masking
        radius: Radius for spherical masking

    Returns:
        Boolean numpy array (True inside sphere, False outside)

    """
    temp_mask = map_grid.clone()
    temp_mask.fill(0)
    temp_mask.set_points_around(
        gemmi.Position(position[0], position[1], position[2]),
        radius=radius,
        value=1,
    )
    temp_mask.symmetrize_max()
    return np.array(temp_mask, copy=False).astype(bool)


def save_mask_as_ccp4(target_ccp4_map, mask_position, mask_radius, output_path: str):
    """Save spherical mask as CCP4 file."""
    mask_grid = target_ccp4_map.grid.clone()
    mask_grid.fill(0)
    mask_grid.set_points_around(
        gemmi.Position(mask_position[0], mask_position[1], mask_position[2]),
        radius=mask_radius,
        value=1,
    )
    mask_grid.symmetrize_max()

    mask_ccp4 = gemmi.Ccp4Map()
    mask_ccp4.grid = mask_grid
    mask_ccp4.update_ccp4_header()
    mask_ccp4.write_ccp4_map(output_path)
    print(f"Mask saved to: {output_path}")


def save_map_as_ccp4(model_map: torch.Tensor, target_ccp4_map, output_path: str):
    """Save torch tensor map as CCP4 file."""
    try:
        model_map_np = model_map.detach().cpu().numpy()

        output_ccp4 = gemmi.Ccp4Map()
        output_ccp4.grid = gemmi.FloatGrid()
        output_ccp4.grid.copy_metadata_from(target_ccp4_map.grid)
        output_ccp4.grid.set_size_without_checking(*model_map_np.shape)
        output_ccp4.grid.set_values(model_map_np.flatten())
        output_ccp4.update_ccp4_header()
        output_ccp4.write_ccp4_map(output_path)
        print(f"Map saved to: {output_path}")

    except Exception as e:
        print(f"Error saving map: {e}")
        fallback_path = output_path.replace(".ccp4", "_numpy.npy")
        np.save(fallback_path, model_map_np)
        print(f"Saved as numpy array: {fallback_path}")
