import glob
import os
import pickle
import re
from pathlib import Path

import numpy as np
import skbio
import torch
from openfold import config as of_config
from openfold.data import data_pipeline, feature_pipeline, templates
from SFC_Torch import PDBParser

import rocket
from rocket import coordinates as rk_coordinates
from rocket import utils as rk_utils


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


def build_processed_features_from_alignment(
    fasta_path: str,
    alignment_dir: str,
    preset: str = "model_1_ptm",
    device: str = "cpu",
    max_recycling_iters: int | None = None,
    template_mmcif_dir: str | None = None,
    kalign_binary_path: str | None = None,
    max_template_date: str | None = None,
    template_max_hits: int | None = None,
    template_release_dates_path: str | None = None,
    template_obsolete_pdbs_path: str | None = None,
):
    if template_mmcif_dir is None:
        template_mmcif_dir = os.environ.get("OPENFOLD_MMCIF_DIR")
        if template_mmcif_dir is None:
            data_dir = os.environ.get("OPENFOLD_DATA_DIR")
            if data_dir:
                for candidate in (
                    "pdb_mmcif",
                    "mmcif",
                    "mmcif_dir",
                    "mmcif_files",
                ):
                    candidate_path = os.path.join(data_dir, candidate)
                    if os.path.isdir(candidate_path):
                        template_mmcif_dir = candidate_path
                        break

    if kalign_binary_path is None:
        kalign_binary_path = os.environ.get("KALIGN_BINARY_PATH")

    template_featurizer = None
    if template_mmcif_dir and kalign_binary_path:
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date=max_template_date or "2100-01-01",
            max_hits=template_max_hits or 20,
            kalign_binary_path=kalign_binary_path,
            release_dates_path=template_release_dates_path,
            obsolete_pdbs_path=template_obsolete_pdbs_path,
        )

    data_processor = data_pipeline.DataPipeline(template_featurizer=template_featurizer)
    raw_features = generate_feature_dict(
        fasta_path=fasta_path,
        alignment_dir=alignment_dir,
        data_processor=data_processor,
    )

    cfg = of_config.model_config(preset)
    if max_recycling_iters is not None:
        cfg.data.common.max_recycling_iters = max_recycling_iters

    feature_processor = feature_pipeline.FeaturePipeline(cfg.data)
    processed_features = feature_processor.process_features(
        raw_features,
        mode="predict",
        is_multimer=False,
    )

    if device != "cpu":
        processed_features = rk_utils.move_tensors_to_device(
            processed_features, device=device
        )

    return processed_features, raw_features, feature_processor


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
    output_pdb_path: str | Path | None = None,
    outputs: dict | None = None,
    raw_feature_dict: dict | None = None,
    feature_processor=None,
    multimer_ri_gap: int = 200,
    subtract_plddt: bool = False,
    processed_feature_dict_override: dict | None = None,
    write_pdb_only: bool = False,
):
    if bias_version != 3:
        raise ValueError("Only bias_version=3 is supported.")

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

    if write_pdb_only:
        if (
            output_pdb_path is None
            or outputs is None
            or raw_feature_dict is None
            or feature_processor is None
            or processed_feature_dict_override is None
        ):
            raise ValueError(
                "output_pdb_path, outputs, raw_feature_dict, feature_processor, "
                "and processed_feature_dict_override are required for write_pdb_only"
            )
        from openfold.np import protein
        from openfold.utils.script_utils import prep_output
        from openfold.utils.tensor_utils import tensor_tree_map

        def _to_numpy_last_recycle(val):
            if torch.is_tensor(val):
                val = val.detach().cpu()
            val = np.array(val)
            return val[..., -1] if val.ndim > 0 else val

        processed_np = tensor_tree_map(
            _to_numpy_last_recycle,
            processed_feature_dict_override,
        )
        out_np = tensor_tree_map(
            lambda x: np.array(x.detach().cpu()) if torch.is_tensor(x) else x,
            outputs,
        )
        unrelaxed_protein = prep_output(
            out_np,
            processed_np,
            raw_feature_dict,
            feature_processor,
            PRESET,
            multimer_ri_gap=multimer_ri_gap,
            subtract_plddt=subtract_plddt,
        )
        output_pdb_path = Path(output_pdb_path)
        output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
        output_pdb_path.write_text(protein.to_pdb(unrelaxed_protein))

        return None, None, None

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
    if bias_version != 3:
        raise ValueError("Only bias_version=3 is supported.")

    num_res = device_processed_features["aatype"].shape[0]
    device_processed_features["msa_feat_bias"] = torch.zeros(
        (512, num_res, 23), requires_grad=True, device=device
    )

    if isinstance(starting_weights, torch.Tensor):
        device_processed_features["msa_feat_weights"] = (
            starting_weights.detach().to(device=device).requires_grad_(True)
        )
    elif starting_weights is not None:
        weight_matches = glob.glob(starting_weights)
        if not weight_matches:
            raise FileNotFoundError(f"No starting weights matched: {starting_weights}")
        device_processed_features["msa_feat_weights"] = (
            torch
            .load(weight_matches[0])
            .detach()
            .to(device=device)
            .requires_grad_(True)
        )
        print("Loaded starting weights from:", weight_matches[0])
    else:
        device_processed_features["msa_feat_weights"] = torch.ones(
            (512, num_res, 23), requires_grad=True, device=device
        )

    if recombination_bias is not None:
        device_processed_features["msa_feat_bias"] = (
            recombination_bias.detach().to(device=device).requires_grad_(True)
        )
    elif isinstance(starting_bias, torch.Tensor):
        device_processed_features["msa_feat_bias"] = (
            starting_bias.detach().to(device=device).requires_grad_(True)
        )
    elif starting_bias is not None:
        bias_matches = glob.glob(starting_bias)
        if not bias_matches:
            raise FileNotFoundError(f"No starting bias matched: {starting_bias}")
        device_processed_features["msa_feat_bias"] = (
            torch.load(bias_matches[0]).detach().to(device=device).requires_grad_(True)
        )
        print("Loaded starting bias from:", bias_matches[0])

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
