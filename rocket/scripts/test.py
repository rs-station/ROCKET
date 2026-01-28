import argparse
import os
import pickle

from openfold.config import model_config
from SFC_Torch import PDBParser

import rocket
from rocket import coordinates as rk_coordinates
from rocket import utils as rk_utils

PRESET = "model_1"


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    parser.add_argument(
        "-p",
        "--path",
        default="/net/cci/alisia/openfold_tests/run_openfold/test_cases",
        help=("Path to the parent folder"),
    )

    # Required arguments
    parser.add_argument(
        "-root",
        "--file_root",
        required=True,
        help=("PDB code or filename root for the dataset"),
    )

    parser.add_argument(
        "-v",
        "--bias_version",
        required=True,
        type=int,
        help=("Bias version to implement (1, 2, 3, 4)"),
    )

    parser.add_argument(
        "-it",
        "--iterations",
        required=True,
        type=int,
        help=("Refinement iterations"),
    )

    # Optional arguments
    parser.add_argument(
        "-target",
        "--target_pdb",
        default=None,
        help=("Name of target pdb file in the file_root"),
    )

    parser.add_argument(
        "-lr_a",
        "--additive_learning_rate",
        type=float,
        default=1e-4,
        help=("Learning rate for additive bias. Default 1e-3"),
    )

    parser.add_argument(
        "-lr_m",
        "--multiplicative_learning_rate",
        type=float,
        default=1e-3,
        help=("Learning rate for multiplicative bias. Default 1e-2"),
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help=("Weight decay used in adamW. Default None, use adam"),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help=("Device to run the model"),
    )

    parser.add_argument(
        "-n",
        "--note",
        type=str,
        default="",
        help=("Optional additional identified"),
    )

    return parser.parse_args()


def mse_optimize(
    path: str,
    file_root: str,
    bias_version: int,
    iterations: int,
    target_pdb: str,
    device: str,
    additive_learning_rate: float,
    multiplicative_learning_rate: float,
    weight_decay: float | None = 0.0001,
    note: str = "",
):
    input_pdb_path = f"{path}/{file_root}/{file_root}-pred-aligned.pdb"
    moving_pdb_obj = PDBParser(input_pdb_path)

    # Model initialization
    version_to_class = {
        1: rocket.MSABiasAFv1,
        2: rocket.MSABiasAFv2,
        3: rocket.MSABiasAFv3,
        4: rocket.TemplateBiasAF,
    }
    af_bias = version_to_class[bias_version](
        model_config(PRESET, train=True), PRESET
    ).to(device)
    af_bias.eval()
    af_bias.freeze()

    output_directory_path = f"{path}/{file_root}/outputs/MSEoptimize_{note}"

    with open(
        f"{path}/{file_root}/{file_root}_processed_feats.pickle",
        "rb",
    ) as file:
        # Load the data from the pickle file
        processed_features = pickle.load(file)

    device_processed_features = rk_utils.move_tensors_to_device(
        processed_features, device=device
    )

    try:
        os.makedirs(output_directory_path, exist_ok=True)
    except FileExistsError:
        print(
            f"Warning: Directory '{output_directory_path}' already exists. Overwriting."
        )

    print("starting prediction")
    af2_output, prevs = af_bias(
        device_processed_features,
        [None, None, None],
        num_iters=20,
        bias=False,
    )
    print("prediction finished")
    prevs = [tensor.detach() for tensor in prevs]

    ### test
    xyz_orth_sfc, plddts = rk_coordinates.extract_allatoms(
        af2_output, device_processed_features, moving_pdb_obj.cra_name
    )
    moving_pdb_obj.set_positions(rk_utils.assert_numpy(xyz_orth_sfc))
    moving_pdb_obj.savePDB(f"{output_directory_path!s}/minimaltesting.pdb")


def main():
    args = parse_arguments()
    args_dict = vars(args)
    mse_optimize(**args_dict)


if __name__ == "__main__":
    main()
