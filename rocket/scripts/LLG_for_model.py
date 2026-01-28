import os
import uuid

from openfold.config import model_config

import rocket
from rocket import coordinates as rk_coordinates
from rocket import refinement_utils as rkrf_utils
from rocket.cryo import structurefactors as cryo_sf
from rocket.cryo import targets as cryo_targets

device = "cuda:0"
PRESET = "model_1"
EXCLUDING_RES = None


target_id = "ap3_d_short"
# path = "/n/hekstra_lab/people/minhuan/projects/AF2_refine/cryoEM_dev/test_systems"
path = "/net/cci/alisia/cryo_rocket"
# mtz_file = f"{path}/{target_id}/{target_id}-Edata_below6.mtz"
mtz_file = f"{path}/{target_id}/{target_id}-Edata_7A_nov28.mtz"
input_pdb = f"{path}/{target_id}/ap3_d_short-pred-aligned-nov28.pdb"
note = "RBR+LLGcalculation"
n_bins = 20
bias_version = 3

refinement_run_uuid = uuid.uuid4().hex
output_directory_path = f"{path}/{target_id}/outputs/{refinement_run_uuid}/{note}"
try:
    os.makedirs(output_directory_path, exist_ok=True)
except FileExistsError:
    print(
        f"Warning: Directory '{output_directory_path}' already exists. Overwriting..."
    )
    print(
        f"System: {target_id}, refinment run ID: {refinement_run_uuid!s}, Note: {note}",
        flush=True,
    )


# Initialize SFC

cryo_sfc = cryo_sf.initial_cryoSFC(
    input_pdb,
    mtz_file,
    "Emean",
    "PHIEmean",
    device,
    n_bins,
)

sfc_rbr = cryo_sf.initial_cryoSFC(
    input_pdb,
    mtz_file,
    "Emean",
    "PHIEmean",
    device,
    n_bins,
)


cra_calphas_list, calphas_mask = rk_coordinates.select_CA_from_craname(
    cryo_sfc.cra_name
)
residue_numbers = [int(name.split("-")[1]) for name in cra_calphas_list]

# LLG initialization
cryo_llgloss = cryo_targets.LLGloss(cryo_sfc, mtz_file)
cryo_llgloss_rbr = cryo_targets.LLGloss(cryo_sfc, mtz_file)


# Model initialization
version_to_class = {
    1: rocket.MSABiasAFv1,
    2: rocket.MSABiasAFv2,
    3: rocket.MSABiasAFv3,
    4: rocket.TemplateBiasAF,
}
af_bias = version_to_class[bias_version](model_config(PRESET, train=True), PRESET).to(
    device
)
af_bias.freeze()


run_id = rkrf_utils.number_to_letter(1)

# Initialize the processed dict space
device_processed_features, feature_key, features_at_it_start = (
    rkrf_utils.init_processed_dict(
        bias_version=bias_version,
        path=path,
        file_root=target_id,
        device=device,
        PRESET=PRESET,
    )
)
# Initialize bias
device_processed_features, optimizer, bias_names = rkrf_utils.init_bias(
    device_processed_features=device_processed_features,
    bias_version=bias_version,
    device=device,
    lr_a=0.0,
    lr_m=0.0,
)

device_processed_features[feature_key] = features_at_it_start.detach().clone()

aligned_xyz = cryo_sfc.atom_pos_orth.clone()

og_llg = -cryo_llgloss(
    aligned_xyz,
)

print(f"LLG for input {input_pdb} model is", og_llg.item())

# Rigid body refinement (RBR) step
optimized_xyz, loss_track_pose = rk_coordinates.rigidbody_refine_quat(
    aligned_xyz, cryo_llgloss_rbr, sfc_rbr.cra_name, lbfgs=True
)


# Calculate (or refine) sigmaA
cryo_llgloss.sfc.atom_pos_orth = optimized_xyz.detach().clone()
cryo_llgloss.sfc.savePDB(f"{output_directory_path!s}/{run_id}_RBR+calculated.pdb")

# LLG loss
L_llg = -cryo_llgloss(
    optimized_xyz,
)

print("LLG for RBR model is", L_llg.item())
