import glob
import pickle
import re

import numpy as np
import skbio
import torch
from SFC_Torch import PDBParser

import rocket
from rocket import coordinates as rk_coordinates
from rocket import utils as rk_utils
from rocket.xtal import utils as llg_utils


def generate_feature_dict(
    fasta_path,
    alignment_dir,
    data_processor,
):
    feature_dict = data_processor.process_fasta(
        fasta_path=fasta_path,
        alignment_dir=alignment_dir,
        seqemb_mode=False,
    )
    return feature_dict


def number_to_letter(n):
    if 0 <= n <= 25:
        return chr(n + 65)
    else:
        return None


def get_identical_indices(A, B):
    """
    Get indices of aligned string A and B to produce identical sequence

    >>> A = 'EWTUY'
    >>> B = 'E-RUY'
    >>> get_identical_indices(A, B)
    [0,3,4], [0,2,3]

    So A[0,3,4] = 'EUY' = B[0,2,3]
    """
    ind_A = []
    ind_B = []
    ai = 0
    bi = 0
    for a, b in zip(A, B, strict=False):
        if a == "-":
            bi += 1
            continue
        if b == "-":
            ai += 1
            continue
        if a == b:
            ind_A.append(ai)
            ind_B.append(bi)
            ai += 1
            bi += 1
        else:
            ai += 1
            bi += 1
    return np.array(ind_A), np.array(ind_B)


def get_pattern_index(str_list, pattern):
    return next((i for i, s in enumerate(str_list) if re.match(pattern, s)), None)


def get_common_ca_ind(pdb1: PDBParser, pdb2: PDBParser):
    """
    A known bug: it can throw some residues out when the two pdbs have ideentical sequences
    for example, "DFGTT" for both, and it will only keep "GTT"
    """  # noqa: E501
    seq1 = pdb1.sequence
    seq2 = pdb2.sequence
    alignment = skbio.alignment.StripedSmithWaterman(seq1)(
        seq2
    )  # Align sequence with Smith Waterman Algorithm
    subind_1 = np.arange(alignment.query_begin, alignment.query_end + 1)
    subind_2 = np.arange(alignment.target_begin, alignment.target_end_optimal + 1)
    subsubind_1, subsubind_2 = get_identical_indices(
        alignment.aligned_query_sequence, alignment.aligned_target_sequence
    )
    common_seq1 = subind_1[subsubind_1]
    common_seq2 = subind_2[subsubind_2]
    common_ca_ind_1 = [
        get_pattern_index(pdb1.cra_name, rf".*-{j}-.*-CA$") for j in common_seq1
    ]
    common_ca_ind_2 = [
        get_pattern_index(pdb2.cra_name, rf".*-{i}-.*-CA$") for i in common_seq2
    ]
    assert (
        np.array([i[-6:] for i in np.array(pdb1.cra_name)[common_ca_ind_1]])
        == np.array([i[-6:] for i in np.array(pdb2.cra_name)[common_ca_ind_2]])
    ).all()
    return common_ca_ind_1, common_ca_ind_2


def get_common_bb_ind(pdb1, pdb2):
    seq1 = pdb1.sequence
    seq2 = pdb2.sequence
    alignment = skbio.alignment.StripedSmithWaterman(seq1)(
        seq2
    )  # Align sequence with Smith Waterman Algorithm
    subind_1 = np.arange(alignment.query_begin, alignment.query_end + 1)
    subind_2 = np.arange(alignment.target_begin, alignment.target_end_optimal + 1)
    subsubind_1, subsubind_2 = get_identical_indices(
        alignment.aligned_query_sequence, alignment.aligned_target_sequence
    )
    common_seq1 = subind_1[subsubind_1]
    common_seq2 = subind_2[subsubind_2]
    common_ca_ind_1 = [
        get_pattern_index(pdb1.cra_name, rf".*-{j}-.*-CA$") for j in common_seq1
    ]
    common_N_ind_1 = [
        get_pattern_index(pdb1.cra_name, rf".*-{j}-.*-N$") for j in common_seq1
    ]
    common_C_ind_1 = [
        get_pattern_index(pdb1.cra_name, rf".*-{j}-.*-C$") for j in common_seq1
    ]
    common_ca_ind_2 = [
        get_pattern_index(pdb2.cra_name, rf".*-{i}-.*-CA$") for i in common_seq2
    ]
    common_N_ind_2 = [
        get_pattern_index(pdb2.cra_name, rf".*-{i}-.*-N$") for i in common_seq2
    ]
    common_C_ind_2 = [
        get_pattern_index(pdb2.cra_name, rf".*-{i}-.*-C$") for i in common_seq2
    ]

    filtered_ca_ind_1 = list(filter(lambda x: x is not None, common_ca_ind_1))
    filtered_N_ind_1 = list(filter(lambda x: x is not None, common_N_ind_1))
    filtered_C_ind_1 = list(filter(lambda x: x is not None, common_C_ind_1))

    filtered_ca_ind_2 = list(filter(lambda x: x is not None, common_ca_ind_2))
    filtered_N_ind_2 = list(filter(lambda x: x is not None, common_N_ind_2))
    filtered_C_ind_2 = list(filter(lambda x: x is not None, common_C_ind_2))

    # Now add only the valid lists
    common_bb_ind_1 = filtered_ca_ind_1 + filtered_N_ind_1 + filtered_C_ind_1
    common_bb_ind_2 = filtered_ca_ind_2 + filtered_N_ind_2 + filtered_C_ind_2
    assert (
        np.array([i[-6:] for i in np.array(pdb1.cra_name)[common_bb_ind_1]])
        == np.array([i[-6:] for i in np.array(pdb2.cra_name)[common_bb_ind_2]])
    ).all()
    return common_bb_ind_1, common_bb_ind_2


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class EarlyStopper:
    def __init__(self, patience=200, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float("inf")

    def early_stop(self, loss):
        if loss < (self.min_loss - self.min_delta):
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def init_processed_dict(
    bias_version,
    path,
    device,
    template_pdb=None,
    target_seq=None,
    PRESET="model_1_ptm",
    postfix="processed_feats.pickle",
    processed_feats_path: str | None = None,
):
    if bias_version == 4:
        device_processed_features = rocket.make_processed_dict_from_template(
            template_pdb=f"{path}/ROCKET_inputs/{template_pdb}",
            target_seq=target_seq,
            config_preset=PRESET,
            device=device,
            msa_dict=None,
        )
        features_at_it_start = (
            device_processed_features["template_torsion_angles_sin_cos"]
            .detach()
            .clone()
        )
        feature_key = "template_torsion_angles_sin_cos"
    else:
        processed_features = None
        if processed_feats_path:
            with open(processed_feats_path, "rb") as file:
                processed_features = pickle.load(file)
        else:
            prediction_glob = glob.glob(f"{path}/predictions/*{postfix}")
            if prediction_glob:
                with open(prediction_glob[0], "rb") as file:
                    processed_features = pickle.load(file)
            else:
                input_glob = glob.glob(f"{path}/*{postfix}")
                if not input_glob:
                    input_glob = glob.glob(f"{path}/*.pickle")
                if not input_glob:
                    raise FileNotFoundError(
                        "No processed features found in predictions/ or input dir."
                    )
                with open(input_glob[0], "rb") as file:
                    processed_features = pickle.load(file)

        device_processed_features = rk_utils.move_tensors_to_device(
            processed_features, device=device
        )
        features_at_it_start = device_processed_features["msa_feat"].detach().clone()
        feature_key = "msa_feat"
    return device_processed_features, feature_key, features_at_it_start


def init_llgloss(sfc, tng_file, min_resolution=None, max_resolution=None):
    resol_min = min(sfc.dHKL) if min_resolution is None else min_resolution
    resol_max = max(sfc.dHKL) if max_resolution is None else max_resolution
    llgloss = rocket.xtal.targets.LLGloss(
        sfc, tng_file, sfc.device, resol_min, resol_max
    )
    return llgloss


def init_bias(
    device_processed_features,
    bias_version,
    device,
    lr_a,
    lr_m,
    weight_decay=None,
    starting_bias=None,
    starting_weights=None,
    recombination_bias=None,
):
    num_res = device_processed_features["aatype"].shape[0]
    device_processed_features["msa_feat_bias"] = torch.zeros(
        (512, num_res, 23), requires_grad=True, device=device
    )

    if bias_version == 4:
        device_processed_features["template_torsion_angles_sin_cos_bias"] = (
            torch.zeros_like(
                device_processed_features["template_torsion_angles_sin_cos"],
                requires_grad=True,
                device=device,
            )
        )
        if weight_decay is None:
            optimizer = torch.optim.Adam([
                {
                    "params": device_processed_features[
                        "template_torsion_angles_sin_cos_bias"
                    ],
                    "lr": lr_a,
                },
            ])
        else:
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": device_processed_features[
                            "template_torsion_angles_sin_cos_bias"
                        ],
                        "lr": lr_a,
                    },
                ],
                weight_decay=weight_decay,
            )
        bias_names = ["template_torsion_angles_sin_cos_bias"]

    elif bias_version == 3:
        if starting_weights is not None:
            device_processed_features["msa_feat_weights"] = (
                torch.load(glob.glob(starting_weights)[0])
                .detach()
                .to(device=device)
                .requires_grad_(True)
            )
        else:
            device_processed_features["msa_feat_weights"] = torch.ones(
                (512, num_res, 23), requires_grad=True, device=device
            )

        if recombination_bias is not None:
            device_processed_features["msa_feat_bias"] = (
                recombination_bias.detach().to(device=device).requires_grad_(True)
            )
        elif starting_bias is not None:
            device_processed_features["msa_feat_bias"] = (
                torch.load(glob.glob(starting_bias)[0])
                .detach()
                .to(device=device)
                .requires_grad_(True)
            )

        if weight_decay is None:
            optimizer = torch.optim.Adam([
                {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                {
                    "params": device_processed_features["msa_feat_weights"],
                    "lr": lr_m,
                },
            ])
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                    {
                        "params": device_processed_features["msa_feat_weights"],
                        "lr": lr_m,
                    },
                ],
                weight_decay=weight_decay,
            )
        bias_names = ["msa_feat_bias", "msa_feat_weights"]

    elif bias_version == 2:
        device_processed_features["msa_feat_weights"] = torch.eye(
            512, dtype=torch.float32, requires_grad=True, device=device
        )

        if weight_decay is None:
            optimizer = torch.optim.Adam([
                {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                {
                    "params": device_processed_features["msa_feat_weights"],
                    "lr": lr_m,
                },
            ])
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                    {
                        "params": device_processed_features["msa_feat_weights"],
                        "lr": lr_m,
                    },
                ],
                weight_decay=weight_decay,
            )
        bias_names = ["msa_feat_bias", "msa_feat_weights"]

    elif bias_version == 1:
        if weight_decay is None:
            optimizer = torch.optim.Adam([
                {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
            ])
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                ],
                weight_decay=weight_decay,
            )

        bias_names = ["msa_feat_bias"]

    return device_processed_features, optimizer, bias_names


def position_alignment(
    af2_output,
    device_processed_features,
    cra_name,
    best_pos,
    exclude_res,
    domain_segs=None,
    reference_bfactor=None,
):
    xyz_orth_sfc, plddts = rk_coordinates.extract_allatoms(
        af2_output, device_processed_features, cra_name
    )
    plddts_res = rk_utils.assert_numpy(af2_output["plddt"])
    pseudo_Bs = rk_utils.plddt2pseudoB_pt(plddts)

    # MH @ Sep 10 2024, temp edits to convert weighted kabsch to cutoff kabsch
    if reference_bfactor is None:
        pseudoB_np = rk_utils.assert_numpy(pseudo_Bs)
        cutoff1 = np.quantile(pseudoB_np, 0.3)
        cutoff2 = cutoff1 * 1.5
        weights = rk_utils.weighting(pseudoB_np, cutoff1, cutoff2)
    else:
        assert reference_bfactor.shape == pseudo_Bs.shape, (
            "Reference bfactor should have same shape as model bfactor!"
        )
        reference_bfactor_np = rk_utils.assert_numpy(reference_bfactor)
        cutoff1 = np.quantile(reference_bfactor_np, 0.3)
        cutoff2 = cutoff1 * 1.5
        weights = rk_utils.weighting(reference_bfactor_np, cutoff1, cutoff2)
    # plddts_np = rk_utils.assert_numpy(plddts)
    # weights = np.ones_like(plddts_np)
    # weights[plddts_np < 85.0] = 1e-5

    aligned_xyz = rk_coordinates.iterative_kabsch_alignment(
        xyz_orth_sfc,
        best_pos,
        cra_name,
        weights=weights,
        exclude_res=exclude_res,
        domain_segs=domain_segs,
    )
    return aligned_xyz, plddts_res, pseudo_Bs.detach()


def update_sigmaA(
    llgloss,
    llgloss_rbr,
    aligned_xyz,
    constant_fp_added_HKL=None,
    constant_fp_added_asu=None,
):
    Ecalc, Fc = llgloss.compute_Ecalc(
        aligned_xyz.detach(),
        return_Fc=True,
        return_Rfactors=False,
        update_scales=True,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
    )
    Ecalc_rbr, _ = llgloss_rbr.compute_Ecalc(
        aligned_xyz.detach(),
        return_Fc=True,
        return_Rfactors=False,
        solvent=False,
        update_scales=True,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
    )
    llgloss.refine_sigmaA_newton(
        Ecalc, n_steps=5, subset="working", smooth_overall_weight=0.0
    )
    llgloss_rbr.refine_sigmaA_newton(
        Ecalc_rbr, n_steps=2, subset="working", smooth_overall_weight=0.0
    )
    return llgloss, llgloss_rbr, Ecalc, Fc


def sigmaA_from_true(
    llgloss,
    llgloss_rbr,
    aligned_xyz,
    Etrue,
    phitrue,
    constant_fp_added_HKL=None,
    constant_fp_added_asu=None,
):
    Ecalc, Fc = llgloss.compute_Ecalc(
        aligned_xyz.detach(),
        return_Fc=True,
        update_scales=True,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
    )
    Ecalc_rbr, Fc_rbr = llgloss_rbr.compute_Ecalc(
        aligned_xyz.detach(),
        return_Fc=True,
        solvent=False,
        update_scales=True,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
    )
    sigmas = llg_utils.sigmaA_from_model(
        Etrue,
        phitrue,
        Ecalc,
        Fc,
        llgloss.sfc.dHKL,
        llgloss.bin_labels,
    )
    llgloss.sigmaAs = sigmas
    sigmas_rbr = llg_utils.sigmaA_from_model(
        Etrue,
        phitrue,
        Ecalc_rbr,
        Fc_rbr,
        llgloss.sfc.dHKL,
        llgloss.bin_labels,
    )
    llgloss_rbr.sigmaAs = sigmas_rbr
    return llgloss, llgloss_rbr
