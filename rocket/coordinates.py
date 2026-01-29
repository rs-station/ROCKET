"""
Functions relating model coordinates/PDB file modificaitons
"""

import time

import numpy as np
import torch
from openfold.np import residue_constants
from scipy.spatial.transform import Rotation
from SFC_Torch import SFcalculator

from rocket import utils


def rigidbody_refine_quat(
    xyz,
    llgloss,
    cra_name,
    lbfgs=False,
    added_chain_HKL=None,
    added_chain_asu=None,
    lbfgs_lr=150.0,
    verbose=True,
    domain_segs=None,
):
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
            llgloss,
            xyz.detach(),
            llgloss.device,
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
    llgloss,
    xyz,
    device,
    domain_bools,
    added_chain_HKL=None,
    added_chain_asu=None,
    lbfgs_lr=150.0,
    verbose=True,
):
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
        llgloss,
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


def find_rigidbody_matrix_adam_quat(
    llgloss,
    propose_com,
    propose_rmcom,
    device,
    added_chain_HKL=None,
    added_chain_asu=None,
    verbose=True,
):
    q = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True
    )
    trans_vec = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
    loss_track_pose = pose_train_adam_quat(
        llgloss,
        q,
        trans_vec,
        propose_com,
        propose_rmcom,
        loss_track=[],
        added_chain_HKL=added_chain_HKL,
        added_chain_asu=added_chain_asu,
        verbose=verbose,
    )
    return trans_vec, q, loss_track_pose


def pose_train_lbfgs_quat(
    llgloss,
    qs,
    trans_vecs,
    xyz,
    domain_bools,
    lr=150.0,
    n_steps=15,
    loss_track=None,
    added_chain_HKL=None,
    added_chain_asu=None,
    verbose=True,
):
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
        loss = -llgloss(
            temp_model,
            bin_labels=None,
            num_batch=1,
            sub_ratio=1.0,
            solvent=False,
            return_Rfactors=False,
            added_chain_HKL=added_chain_HKL,
            added_chain_asu=added_chain_asu,
        )
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


def pose_train_adam_quat(
    llgloss,
    q,
    trans_vec,
    propose_com,
    propose_rmcom,
    lr=1e-3,
    n_steps=100,
    loss_track=None,
    added_chain_HKL=None,
    added_chain_asu=None,
    verbose=True,
):
    if loss_track is None:
        loss_track = []

    def pose_steptrain(optimizer):
        optimizer.zero_grad()
        temp_R = quaternions_to_SO3(q)
        temp_model = torch.matmul(propose_rmcom, temp_R) + propose_com + trans_vec
        loss = -llgloss(
            temp_model,
            bin_labels=None,
            num_batch=1,
            sub_ratio=1.0,
            solvent=False,
            return_Rfactors=False,
            added_chain_HKL=added_chain_HKL,
            added_chain_asu=added_chain_asu,
        )
        loss.backward()
        optimizer.step()
        return loss.item()

    start_time = time.time()
    optimizer = torch.optim.Adam([q, trans_vec], lr=lr)
    for _ in range(n_steps):
        temp = pose_steptrain(optimizer)
        loss_track.append(temp)
    elapsed_time = time.time() - start_time
    if verbose:
        print(
            f"Adam RBR, {n_steps} steps, time taken: {elapsed_time:.4f} seconds",
            flush=True,
        )
    return loss_track


def pose_train_adam_matrix(
    llgloss,
    rot_v1,
    rot_v2,
    trans_vec,
    propose_com,
    propose_rmcom,
    lr=1e-3,
    n_steps=100,
    loss_track=None,
    added_chain=None,
):
    if loss_track is None:
        loss_track = []

    def pose_steptrain(optimizer):
        optimizer.zero_grad()
        temp_R = construct_SO3(rot_v1, rot_v2)
        temp_model = torch.matmul(propose_rmcom, temp_R) + propose_com + trans_vec
        loss = -llgloss(
            temp_model,
            bin_labels=None,
            num_batch=1,
            return_Rfactors=False,
            sub_ratio=1.0,
            solvent=False,
            added_chain=added_chain,
        )
        loss.backward()
        optimizer.step()
        return loss.item()

    start_time = time.time()
    optimizer = torch.optim.Adam([rot_v1, rot_v2, trans_vec], lr=lr)
    for k in range(n_steps):
        temp = pose_steptrain(optimizer)
        loss_track.append(temp)

        elapsed_time = time.time() - start_time
        print(f"Step {k + 1}/{n_steps} - Time: {elapsed_time:.4f} s, Loss: {temp:.3f}")

    return loss_track


def pose_train_lbfgs(
    llgloss,
    rot_v1,
    rot_v2,
    trans_vec,
    propose_com,
    propose_rmcom,
    lr=0.005,
    n_steps=50,
    loss_track=None,
):
    if loss_track is None:
        loss_track = []

    def closure():
        optimizer.zero_grad()
        temp_R = construct_SO3(rot_v1, rot_v2)
        temp_model = torch.matmul(propose_rmcom, temp_R) + propose_com + trans_vec
        loss = -llgloss(
            temp_model,
            bin_labels=None,
            return_Rfactors=False,
            num_batch=1,
            sub_ratio=1.0,
        )
        loss.backward()
        return loss

    start_time = time.time()

    optimizer = torch.optim.LBFGS(
        [rot_v1, rot_v2, trans_vec],
        lr=lr,
        line_search_fn="strong_wolfe",
        tolerance_change=1e-3,
        max_iter=1,
    )

    for k in range(n_steps):
        temp = optimizer.step(closure)
        loss_track.append(temp.item())
        elapsed_time = time.time() - start_time
        print(f"Step {k + 1}/{n_steps} - Time by optimizer: {elapsed_time:.4f} s")
    return loss_track


def find_rigidbody_matrix_adam(
    llgloss, propose_com, propose_rmcom, device, added_chain
):
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    unit_R = quaternions_to_SO3(q)
    v1, v2 = decompose_SO3(unit_R)
    rot_v1 = torch.tensor(utils.assert_numpy(v1), device=device, requires_grad=True)
    rot_v2 = torch.tensor(utils.assert_numpy(v2), device=device, requires_grad=True)
    trans_vec = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)

    loss_track_pose = pose_train_adam_matrix(
        llgloss,
        rot_v1,
        rot_v2,
        trans_vec,
        propose_com,
        propose_rmcom,
        loss_track=[],
        added_chain=added_chain,
    )
    return trans_vec, rot_v1, rot_v2, loss_track_pose


def find_rigidbody_matrix_lbfgs(llgloss, propose_com, propose_rmcom, device):
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    unit_R = quaternions_to_SO3(q)
    v1, v2 = decompose_SO3(unit_R)
    rot_v1 = torch.tensor(utils.assert_numpy(v1), device=device, requires_grad=True)
    rot_v2 = torch.tensor(utils.assert_numpy(v2), device=device, requires_grad=True)
    trans_vec = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)

    loss_track_pose = pose_train_lbfgs(
        llgloss,
        rot_v1,
        rot_v2,
        trans_vec,
        propose_com,
        propose_rmcom,
        loss_track=[],
    )
    return trans_vec, rot_v1, rot_v2, loss_track_pose


def construct_SO3(v1, v2):
    """
    Construct a continuous representation of SO(3) rotation with two 3D vectors
    https://arxiv.org/abs/1812.07035
    Parameters
    ----------
    v1, v2: 3D tensors
        Real-valued tensor in 3D space
    Returns
    -------
    R, A 3*3 SO(3) rotation matrix
    """
    e1 = v1 / torch.norm(v1)
    u2 = v2 - e1 * torch.tensordot(e1, v2, dims=1)
    e2 = u2 / torch.norm(u2)
    e3 = torch.cross(e1, e2)
    R = torch.stack((e1, e2, e3)).T
    return R


def decompose_SO3(R, a=1, b=1, c=1):
    """
    Decompose the rotation matrix into the two vector representation
    This decomposition is not unique
    a, b, c can be set as arbitrary constants you like
    C != 0
    Parameters
    ----------
    R: 3*3 tensors
        Real-valued rotation matrix
    Returns
    -------
    v1, v2: Two real-valued 3D tensors, continuous representation of the rotation matrix
    """
    assert c != 0, "Give a nonzero c!"
    v1 = a * R[:, 0]
    v2 = b * R[:, 0] + c * R[:, 1]

    return v1, v2


def quaternions_to_SO3(q):
    """
    Normalizes q and maps to group matrix.
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    """
    # q = assert_tensor(q, torch.float32)
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack(
        [
            1 - 2 * j * j - 2 * k * k,
            2 * (i * j - r * k),
            2 * (i * k + r * j),
            2 * (i * j + r * k),
            1 - 2 * i * i - 2 * k * k,
            2 * (j * k - r * i),
            2 * (i * k - r * j),
            2 * (j * k + r * i),
            1 - 2 * i * i - 2 * j * j,
        ],
        -1,
    ).view(*q.shape[:-1], 3, 3)


def select_CA_elements(data):
    return [element.endswith("-CA") for element in data]


def select_CA_from_craname(cra_name_list):
    boolean_mask = select_CA_elements(cra_name_list)
    cra_CAs_list = [
        element
        for element, mask in zip(cra_name_list, boolean_mask, strict=False)
        if mask
    ]
    return cra_CAs_list, boolean_mask


def calculate_mse_loss_per_residue(tensor1, tensor2, residue_numbers):
    mse_losses = []

    for residue in set(residue_numbers):
        # Find indices of atoms with the current residue number in tensor1
        indices1 = [i for i, x in enumerate(residue_numbers) if x == residue]

        if len(indices1) > 0:
            # Extract coordinates for atoms with the current residue number in tensor1
            coords1 = tensor1[indices1, :]

            # Extract coordinates for atoms with the current residue number in tensor2
            coords2 = tensor2[indices1, :]

            # Calculate MSE loss for coordinates of atoms with the same residue number
            mse_loss = torch.sqrt(torch.sum((coords1 - coords2) ** 2))
            mse_losses.append(mse_loss.item())

    return mse_losses


def write_pdb_with_positions(input_pdb_file, positions, output_pdb_file):
    # positions here expected to be rounded to 3 decimal points

    with open(input_pdb_file) as f_in, open(output_pdb_file, "w") as f_out:
        for line in f_in:
            if line.startswith("ATOM"):
                atom_info = line[
                    :30
                ]  # Extract the first 30 characters containing atom information
                rounded_pos = positions.pop(
                    0
                )  # Pop the first rounded position from the list
                new_line = (
                    f"{atom_info}{rounded_pos[0]:8.3f}{rounded_pos[1]:8.3f}{rounded_pos[2]:8.3f}"
                    + line[54:]
                )
                f_out.write(new_line)
            else:
                f_out.write(line)


def fractionalize_torch(atom_pos_orth, unitcell, device=None):
    """
    Apply symmetry operations to real space asu model coordinates

    Parameters
    ----------
    atom_pos_orth: tensor, [N_atom, 3]
        ASU model ccordinates

    Will return fractional coordinates; Otherwise will return orthogonal coordinates

    Return
    ------
    atom_pos_sym_oped, [N_atoms, N_ops, 3] tensor, fractional or orthogonal coordinates
    """
    if device is None:
        device = utils.try_gpu()
    atom_pos_orth.to(device=device)
    orth2frac_tensor = torch.tensor(
        unitcell.fractionalization_matrix.tolist(), device=device
    )
    atom_pos_frac = torch.tensordot(atom_pos_orth, orth2frac_tensor.T, 1)

    return atom_pos_frac


def extract_allatoms(outputs, feats, cra_name_sfc: list):
    N_atom_types = len(residue_constants.atom_types)  # by default 37
    atom_mask = outputs["final_atom_mask"]  # shape [n_res, N_atom_types]
    n_res = atom_mask.shape[0]

    # get atom positions in vectorized manner
    positions_atom = outputs["final_atom_positions"][atom_mask == 1.0]  # [n_atom, 3]

    # get plddt in vectorized manner
    plddt_atom = (
        outputs["plddt"].reshape([-1, 1]).repeat([1, N_atom_types])[atom_mask == 1.0]
    )  # shape [n_atom,]

    # get cra_name from AF2, [chain-resid-resname-atomname,...]
    res_names = utils.assert_numpy(
        [i + "-" for i in list(residue_constants.restype_1to3.values())] + ["UNK-"]
    )
    aatype = feats[
        "aatype"
    ]  # TODO: tackle the match between UNK and real non-standard aa name from SFC
    aatype_1d = res_names[utils.assert_numpy(aatype[:, 0], arr_type=int)]
    chain_resid = np.array([
        "A-" + str(i) + "-" for i in range(n_res)
    ])  # TODO: here we assume all residues in same chain A
    crname_repeats = (
        np.char.add(chain_resid, aatype_1d).reshape(-1, 1).repeat(N_atom_types, axis=-1)
    )  # [n_res, N_atom_types]
    crname_atom = crname_repeats[utils.assert_numpy(atom_mask) == 1]
    atom_types_repeats = (
        utils
        .assert_numpy(residue_constants.atom_types)
        .reshape(1, N_atom_types)
        .repeat(n_res, axis=0)
    )  # [n_res, N_atom_types]
    aname_atom = atom_types_repeats[utils.assert_numpy(atom_mask) == 1]
    cra_name_af = np.char.add(crname_atom, aname_atom).tolist()

    # reorder and assert the same topology
    reorder_index = utils.assert_numpy(
        [cra_name_af.index(i) for i in cra_name_sfc], arr_type=int
    )

    assert np.all(
        utils.assert_numpy(cra_name_af)[reorder_index]
        == utils.assert_numpy(cra_name_sfc)
    ), "Mismatch topolgy between AF and SFC!"

    return positions_atom[reorder_index], plddt_atom[reorder_index]


def extract_atoms_and_backbone(outputs, feats):
    atom_types = residue_constants.atom_types
    atom_mask = outputs["final_atom_mask"]
    pdb_lines = []
    aatype = feats["aatype"]
    atom_positions = outputs["final_atom_positions"]
    selected_atoms_mask = []

    n = aatype.shape[0]
    for i in range(n):
        for atom_name, pos, mask in zip(
            atom_types, atom_positions[i], atom_mask[i], strict=False
        ):
            if mask < 0.5:
                continue
            pdb_lines.append(pos)

            if atom_name in ["C", "CA", "O", "N"]:
                selected_atoms_mask.append(torch.tensor(1, dtype=torch.bool))
            else:
                selected_atoms_mask.append(torch.tensor(0, dtype=torch.bool))

    return torch.stack(pdb_lines), torch.stack(selected_atoms_mask)


def extract_bfactors(prot):
    atom_mask = prot.atom_mask
    aatype = prot.aatype
    b_factors = prot.b_factors

    b_factor_lines = []

    n = aatype.shape[0]
    # Add all atom sites.
    for i in range(n):
        for mask, b_factor in zip(atom_mask[i], b_factors[i], strict=False):
            if mask < 0.5:
                continue

            b_factor_lines.append(b_factor)

    return np.array(b_factor_lines)


def kabsch_align_matrices(tensor1, tensor2):
    # Center the atoms by subtracting their centroids
    centroid1 = torch.mean(tensor1, dim=0, keepdim=True)
    tensor1_centered = tensor1 - centroid1
    centroid2 = torch.mean(tensor2, dim=0, keepdim=True)
    tensor2_centered = tensor2 - centroid2

    # Calculate the covariance matrix
    covariance_matrix = torch.matmul(tensor2_centered.t(), tensor1_centered)

    # Perform Singular Value Decomposition (SVD) on the covariance matrix
    U, _, Vt = torch.linalg.svd(covariance_matrix)

    # Calculate the rotation matrix
    rotation_matrix = torch.matmul(U, Vt)

    # Ensure the determinant is positive
    if torch.det(rotation_matrix) < 0:
        Vt[:, -1] *= -1
        rotation_matrix = torch.matmul(Vt.t(), U.t())

    # Ensure the rotation matrix is not a reflection
    if torch.det(rotation_matrix) < 0:
        rotation_matrix[:, 0] *= -1

    return centroid1, centroid2, rotation_matrix


def select_confident_atoms(current_pos, target_pos, bfacts=None, b_thresh=400.0):
    if bfacts is None:
        # If bfacts is None, set mask to all True
        reshaped_mask = torch.ones_like(current_pos, dtype=torch.bool)
    else:
        # Boolean mask for confident atoms
        mask = bfacts < b_thresh
        reshaped_mask = mask.unsqueeze(1).expand_as(current_pos)

    # Select confident atoms using the mask
    current_pos_conf = torch.flatten(current_pos)[torch.flatten(reshaped_mask)]
    target_pos_conf = torch.flatten(target_pos)[torch.flatten(reshaped_mask)]

    N = current_pos_conf.numel() // 3

    return current_pos_conf.view(N, 3), target_pos_conf.view(N, 3)


def align_tensors(tensor1, centroid1, centroid2, rotation_matrix):
    """
    Apply rotation and translation to a tensor
    """
    tensor1_centered = tensor1 - centroid1
    # Apply the rotation and translation to align the first tensor to the second one
    aligned_tensor1 = torch.matmul(tensor1_centered, rotation_matrix.T) + centroid2

    return aligned_tensor1


def weighted_kabsch_svd(P, Q, weights=None):
    """
    Computes the optimal rotation matrix that minimizes the weighted RMSD
    between two sets of corresponding points P and Q using SVD.

    Args:
        P (np.ndarray): Reference points, shape (N, 3).
        Q (np.ndarray): Points to align to P, shape (N, 3).
        weights (np.ndarray, optional): Weights for each point, shape (N,).

    Returns:
        np.ndarray: Optimal rotation matrix (3, 3).
    """
    if P.shape[0] < 3:
        raise ValueError("Need at least 3 points for stable alignment")

    # Center the point clouds using weighted average
    centroid_P = np.average(P, axis=0, weights=weights)
    centroid_Q = np.average(Q, axis=0, weights=weights)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Compute weighted covariance matrix
    H = Q_centered.T @ np.diag(weights if weights is not None else 1) @ P_centered

    # Perform SVD
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return np.identity(3)  # Fallback for unstable SVD

    # Calculate rotation matrix R
    R = Vt.T @ U.T

    # Handle reflection case (ensures a right-handed coordinate system)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def iterative_kabsch_alignment(
    moving_tensor,
    ref_tensor,
    cra_name,
    weights=None,
    exclude_res=None,
    domain_segs=None,
    cutoff=2.0,
    cycles=5,
):
    """
    Performs Kabsch alignment with iterative outlier rejection on protein structures.

    Args:
        moving_tensor (torch.Tensor): Coords to move, shape [n_points, 3].
        ref_tensor (torch.Tensor): Reference coords, shape [n_points, 3].
        cra_name (List[str]): Chain-residue-atom identifiers, [n_points].
        weights (torch.Tensor|np.ndarray, optional): Weights for the alignment.
        exclude_res (List[int], optional): Residue IDs to exclude.
        domain_segs (List[int], optional): Residue IDs defining domain boundaries.
        cutoff (float): Distance cutoff in Angstroms for outlier rejection.
        cycles (int): Max number of refinement cycles. Set to 0 for a single
                      alignment without outlier rejection.

    Returns:
        torch.Tensor: The aligned moving_tensor, shape [n_points, 3].
    """
    # --- 1. Initial Filtering Setup ---
    backbone_bool = np.array([i.split("-")[-1] in ["N", "CA", "C"] for i in cra_name])
    resid = np.array([int(i.split("-")[1]) for i in cra_name])
    minid, maxid = resid.min(), resid.max()

    if exclude_res is None:
        residue_bool = (resid > minid + 4) & (resid < maxid - 4)
    else:
        residue_bool = np.isin(resid, exclude_res, invert=True)

    # --- 2. Define Domains ---
    if domain_segs is None:
        domain_ranges = [[minid, maxid + 1]]
    else:
        domain_ranges = []
        start = minid
        sorted_segs = sorted(domain_segs)
        for i, seg in enumerate(sorted_segs):
            domain_ranges.append([start, seg])
            start = seg
            if i == len(sorted_segs) - 1:
                domain_ranges.append([start, maxid + 1])

    # --- 3. Process Each Domain ---
    aligned_pos = moving_tensor.clone()
    for domain_start, domain_end_notin in domain_ranges:
        domain_bool = (resid >= domain_start) & (resid < domain_end_notin)
        working_set_mask = backbone_bool & residue_bool & domain_bool

        if not working_set_mask.any():
            continue

        moving_align_coords = utils.assert_numpy(moving_tensor)[working_set_mask]
        ref_align_coords = utils.assert_numpy(ref_tensor)[working_set_mask]
        if moving_align_coords.shape[0] < 3:
            continue

        align_weights = (
            utils.assert_numpy(weights)[working_set_mask]
            if weights is not None
            else None
        )  # noqa: E501

        # --- 4. Iterative Alignment for the Current Domain ---
        inlier_indices = np.arange(moving_align_coords.shape[0])
        final_R = np.identity(3)

        for cycle in range(cycles + 1):
            P_iter = ref_align_coords[inlier_indices]
            Q_iter = moving_align_coords[inlier_indices]
            w_iter = (
                align_weights[inlier_indices] if align_weights is not None else None
            )  # noqa: E501
            try:
                R_iter = weighted_kabsch_svd(P_iter, Q_iter, weights=w_iter)
            except ValueError:
                break
            final_R = R_iter

            if cycle == cycles:
                break

            # Apply transform and find new inliers
            centroid_Q = np.average(Q_iter, axis=0, weights=w_iter)
            centroid_P = np.average(P_iter, axis=0, weights=w_iter)
            t_iter = centroid_P - final_R @ centroid_Q

            transformed_align_coords = (final_R @ moving_align_coords.T).T + t_iter
            distances = np.linalg.norm(
                ref_align_coords - transformed_align_coords, axis=1
            )  # noqa: E501
            new_inlier_indices = np.where(distances < cutoff)[0]

            if len(new_inlier_indices) < 3 or np.array_equal(
                inlier_indices, new_inlier_indices
            ):  # noqa: E501
                break
            inlier_indices = new_inlier_indices
        # --- 5. Apply Final Transformation to the Entire Domain ---
        P_final = ref_align_coords[inlier_indices]
        Q_final = moving_align_coords[inlier_indices]
        w_final = align_weights[inlier_indices] if align_weights is not None else None

        # Calculate final centroids based on the final set of inliers
        final_centroid_moving = np.average(Q_final, axis=0, weights=w_final)
        final_centroid_ref = np.average(P_final, axis=0, weights=w_final)
        # Get all points in the domain and convert components to tensors
        moving_domain_tensor = moving_tensor[domain_bool]
        R_torch = torch.tensor(
            final_R, dtype=moving_tensor.dtype, device=moving_tensor.device
        )
        centroid1_torch = torch.tensor(
            final_centroid_moving,
            dtype=moving_tensor.dtype,
            device=moving_tensor.device,
        )
        centroid2_torch = torch.tensor(
            final_centroid_ref, dtype=moving_tensor.dtype, device=moving_tensor.device
        )

        # Use your function to apply the final transformation
        aligned_domain_tensor = align_tensors(
            moving_domain_tensor, centroid1_torch, centroid2_torch, R_torch
        )
        aligned_pos[domain_bool] = aligned_domain_tensor

    return aligned_pos


def weighted_kabsch(
    moving_tensor,
    ref_tensor,
    cra_name,
    weights=None,
    exclude_res=None,
    domain_segs=None,
):
    """
    Weighted Kabsch Alignment, using scipy implementation
    'scipy.spatial.transform.Rotation.align_vectors'

    Args:
        moving_tensor: torch.Tensor, [n_points, 3]
            coordinates you want to move

        ref_tensor: torch.Tensor, [n_points, 3]
            reference coordinates you want to align to

        cra_name: List[str], [n_points]
            chain-residue-atom name of each atom

        weights: torch.Tensor | np.ndarray, [n_points]
            weights used in the Kabsch Alignment

        exclude_res: List[int] or None
            list of resid you want to exclude from the alignment

        domain_segs: List[int] or None
            List of resid as boundary between different domains,
            i.e. domain_segs = [196] means there are two domains, [0-195] and [196-END]

    Returns:
        aligned_tensor: torch.Tensor, [n_points, 3]
    """
    # use only backbone atoms
    backbone_bool = np.array([i.split("-")[-1] in ["N", "CA", "C"] for i in cra_name])

    # exclude some residues, by default 5 residues on both ends
    resid = [int(i.split("-")[1]) + 1 for i in cra_name]
    minid = min(resid)
    maxid = max(resid)
    if exclude_res is None:
        residue_bool = np.array([(i > minid + 4) and (i < (maxid - 4)) for i in resid])
    else:
        residue_bool = np.array([i not in exclude_res for i in resid])

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
    aligned_pos = torch.ones_like(moving_tensor)
    for domain_range in domain_ranges:
        domain_start, domain_end_notin = domain_range
        domain_bool = np.array([
            (i >= domain_start) and (i < domain_end_notin) for i in resid
        ])
        working_set = backbone_bool & residue_bool & domain_bool
        moving_tensor_np = utils.assert_numpy(moving_tensor)[working_set]
        ref_tensor_np = utils.assert_numpy(ref_tensor)[working_set]
        if weights is None:
            weights_np = None
        else:
            weights_np = utils.assert_numpy(weights)[working_set]

        com_moving = np.average(moving_tensor_np, axis=0, weights=weights_np)
        com_ref = np.average(ref_tensor_np, axis=0, weights=weights_np)
        C, _ = Rotation.align_vectors(
            ref_tensor_np - com_ref, moving_tensor_np - com_moving, weights=weights_np
        )

        rotation_matrix = torch.tensor(C.as_matrix()).to(moving_tensor)
        centroid1 = torch.tensor(com_moving).to(moving_tensor)
        centroid2 = torch.tensor(com_ref).to(moving_tensor)
        aligned_pos[domain_bool] = align_tensors(
            moving_tensor[domain_bool], centroid1, centroid2, rotation_matrix
        )
    return aligned_pos


def cutoff_kabsch(
    moving_tensor, ref_tensor, cra_name, pseudoB, threshB=None, exclude_res=None
):
    # use only backbone atoms
    backbone_bool = np.array([i.split("-")[-1] in ["N", "CA", "C"] for i in cra_name])

    # exclude some residues, by default 5 residues on both ends
    resid = [int(i.split("-")[1]) for i in cra_name]
    if exclude_res is None:
        minid = min(resid)
        maxid = max(resid)
        residue_bool = np.array([(i > minid + 4) and (i < (maxid - 4)) for i in resid])
    else:
        residue_bool = np.array([i not in exclude_res for i in resid])

    if threshB is None:
        bfactor_bool = np.ones_like(backbone_bool, dtype=bool)
    else:
        bfactor_bool = pseudoB < threshB

    working_set = backbone_bool & residue_bool & bfactor_bool
    centroid1, centroid2, rotation_matrix = kabsch_align_matrices(
        moving_tensor[working_set].detach(), ref_tensor[working_set].detach()
    )
    aligned_pos = align_tensors(moving_tensor, centroid1, centroid2, rotation_matrix)

    return aligned_pos


def set_new_positions(orth_pos, frac_pos, sfmodel, device=None):
    if device is None:
        device = utils.try_gpu()
    sfmodel.atom_pos_orth = torch.squeeze(orth_pos, dim=1).to(device)
    sfmodel.atom_pos_frac = torch.squeeze(frac_pos, dim=1).to(device)
    return sfmodel


def transfer_positions(aligned_pos, sfcalculator_model, device=None):
    if device is None:
        device = utils.try_gpu()
    # Transfer positions to sfcalculator
    frac_pos = fractionalize_torch(
        aligned_pos,
        sfcalculator_model.unit_cell,
        sfcalculator_model.space_group,
        device=device,
    )
    sfcalculator_model = set_new_positions(
        aligned_pos, frac_pos, sfcalculator_model, device=device
    )

    # add bfactor calculation based on plddt
    # sfcalculator_model.atom_b_iso = b_factors

    sfcalculator_model = update_sfcalculator(sfcalculator_model)
    return sfcalculator_model


def update_sfcalculator(sfmodel):
    sfmodel.inspect_data(verbose=False)
    sfmodel.calc_fprotein()
    sfmodel.calc_fsolvent()
    sfmodel.init_scales(requires_grad=True)
    sfmodel.calc_ftotal()
    return sfmodel


def initialize_model_frac_pos(model_file, tng_file, device=None):
    if device is None:
        device = utils.try_gpu()
    sfcalculator_model = SFcalculator(
        model_file,
        tng_file,
        expcolumns=["FP", "SIGFP"],
        set_experiment=True,
        testset_value=0,
        device=device,
    )
    target_pos = sfcalculator_model.atom_pos_orth
    sfcalculator_model.atom_pos_frac = sfcalculator_model.atom_pos_frac * 0.00
    sfcalculator_model.atom_pos_orth = sfcalculator_model.atom_pos_orth * 0.00
    sfcalculator_model.atom_pos_frac.requires_grad = True

    return sfcalculator_model, target_pos
