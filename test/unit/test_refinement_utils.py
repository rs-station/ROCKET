from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from rocket.refinement_utils import (
    EarlyStopper,
    generate_feature_dict,
    get_common_bb_ind,
    get_common_ca_ind,
    get_current_lr,
    get_identical_indices,
    get_pattern_index,
    init_bias,
    init_llgloss,
    init_processed_dict,
    number_to_letter,
    position_alignment,
    sigmaA_from_true,
    update_sigmaA,
)


@pytest.fixture
def mock_data_processor():
    """Create a mock data processor object with process_fasta method."""
    data_processor = MagicMock()
    data_processor.process_fasta.return_value = {"features": "test_features"}
    return data_processor


def test_generate_feature_dict(mock_data_processor):
    expected_result = {"features": "test_features"}

    fasta_path = "/path/to/fasta"
    alignment_dir = "/path/to/alignment"

    result = generate_feature_dict(
        fasta_path=fasta_path,
        alignment_dir=alignment_dir,
        data_processor=mock_data_processor,
    )

    # Verify process_fasta was called with correct parameters
    mock_data_processor.process_fasta.assert_called_once_with(
        fasta_path=fasta_path, alignment_dir=alignment_dir, seqemb_mode=False
    )
    assert result == expected_result


def test_number_to_letter():
    # Test valid cases (0-25 should map to A-Z)
    assert number_to_letter(0) == "A"
    assert number_to_letter(1) == "B"
    assert number_to_letter(25) == "Z"

    # Test boundary cases
    assert number_to_letter(-1) is None
    assert number_to_letter(26) is None

    # Test some middle values
    assert number_to_letter(7) == "H"
    assert number_to_letter(12) == "M"
    assert number_to_letter(19) == "T"


def test_get_identical_indices():
    # Test case from docstring
    A = "EWTUY"
    B = "E-RUY"
    ind_A, ind_B = get_identical_indices(A, B)
    assert ind_A.tolist() == [0, 3, 4]
    assert ind_B.tolist() == [0, 2, 3]

    # Test with identical strings
    A = "ABCDE"
    B = "ABCDE"
    ind_A, ind_B = get_identical_indices(A, B)
    assert ind_A.tolist() == [0, 1, 2, 3, 4]
    assert ind_B.tolist() == [0, 1, 2, 3, 4]

    # Test with no identical characters
    A = "ABCDE"
    B = "FGHIJ"
    ind_A, ind_B = get_identical_indices(A, B)
    assert ind_A.tolist() == []
    assert ind_B.tolist() == []

    # Test with gaps in both sequences
    A = "A-BC-DE"
    B = "-AB-CDE"
    ind_A, ind_B = get_identical_indices(A, B)
    assert ind_A.tolist() == [1, 3, 4]
    assert ind_B.tolist() == [1, 3, 4]

    # Test with empty strings
    A = ""
    B = ""
    ind_A, ind_B = get_identical_indices(A, B)
    assert ind_A.tolist() == []
    assert ind_B.tolist() == []


def test_get_pattern_index():
    # Test with more complex patterns
    pdb_like_list = ["A-1-ASP-CA", "A-2-GLY-N", "A-2-GLY-CA", "A-3-PHE-CA"]
    # Find the CA atom of residue 2
    pattern = r".*-2-.*-CA$"
    assert get_pattern_index(pdb_like_list, pattern) == 2

    # Find any atom of PHE
    pattern = r".*-PHE-.*"
    assert get_pattern_index(pdb_like_list, pattern) == 3


@pytest.fixture
def mock_pdb1():
    """Simple mock PDBParser for first protein."""
    pdb = MagicMock()
    pdb.sequence = "GFTT"
    pdb.cra_name = [
        "A-0-GLY-CA",
        "A-1-PHE-CA",
        "A-2-THR-CA",
        "A-3-THR-CA",
        "A-3-THR-N",
    ]
    return pdb


@pytest.fixture
def mock_pdb2():
    """Simple mock PDBParser for second protein."""
    pdb = MagicMock()
    pdb.sequence = "DGFTT"
    pdb.cra_name = [
        "A-0-ASP-CA",
        "A-1-GLY-CA",
        "A-2-PHE-CA",
        "A-3-THR-CA",
        "A-4-THR-CA",
        "A-4-THR-N",
    ]
    return pdb


def test_get_common_ca_ind(mock_pdb1, mock_pdb2):
    """Minimal test to verify the I/O pipeline of get_common_ca_ind."""

    # Call the function
    common_ind1, common_ind2 = get_common_ca_ind(mock_pdb1, mock_pdb2)

    # Verify the function returns numpy arrays
    assert isinstance(common_ind1, list)
    assert isinstance(common_ind2, list)

    assert common_ind1 == [0, 1, 2, 3]
    assert common_ind2 == [1, 2, 3, 4]


def test_get_common_bb_ind(mock_pdb1, mock_pdb2):
    """Minimal test to verify the I/O pipeline of get_common_ca_ind."""

    # Call the function
    common_ind1, common_ind2 = get_common_bb_ind(mock_pdb1, mock_pdb2)

    # Verify the function returns numpy arrays
    assert isinstance(common_ind1, list)
    assert isinstance(common_ind2, list)

    assert common_ind1 == [0, 1, 2, 3, 4]
    assert common_ind2 == [1, 2, 3, 4, 5]


def test_get_current_lr():
    """Test that get_current_lr returns the learning rate from optimizer."""
    # Create a mock optimizer with param_groups
    optimizer = MagicMock()

    # Set up the mock to have param_groups with a learning rate
    expected_lr = 0.001
    optimizer.param_groups = [{"lr": expected_lr}]

    # Test that the function returns the expected learning rate
    result = get_current_lr(optimizer)
    assert result == expected_lr

    # Test with a different learning rate
    new_lr = 0.0001
    optimizer.param_groups = [{"lr": new_lr}]
    result = get_current_lr(optimizer)
    assert result == new_lr


def test_early_stopper():
    """Test basic functionality of EarlyStopper."""
    # Create EarlyStopper with small patience for testing
    stopper = EarlyStopper(patience=2, min_delta=0.1)

    # First call should set the minimum loss and return False (don't stop)
    assert stopper.early_stop(10.0) is False
    assert stopper.min_loss == 10.0
    assert stopper.counter == 0

    # Significant improvement should reset counter and return False
    assert stopper.early_stop(9.8) is False
    assert stopper.min_loss == 9.8
    assert stopper.counter == 0

    # No significant improvement should increment counter and return False
    assert stopper.early_stop(9.75) is False
    assert stopper.min_loss == 9.8  # Unchanged
    assert stopper.counter == 1

    # Still no improvement - should hit patience limit and return True (stop)
    assert stopper.early_stop(9.85) is True
    assert stopper.counter == 2


@patch("rocket.make_processed_dict_from_template")
@patch("glob.glob")
@patch("builtins.open")
@patch("pickle.load")
def test_init_processed_dict(mock_pickle_load, mock_open, mock_glob, mock_make_dict):
    """Test init_processed_dict with both bias_version=4 and other values."""
    # Mock device
    device = torch.device("cpu")

    # Test with bias_version=4 (uses make_processed_dict_from_template)
    # Setup mock return value for rocket.make_processed_dict_from_template
    mock_processed_dict = {"template_torsion_angles_sin_cos": torch.ones(5, 4, 2)}
    mock_make_dict.return_value = mock_processed_dict

    # Call the function
    result_dict, feature_key, features = init_processed_dict(
        bias_version=4,
        path="/test/path",
        device=device,
        template_pdb="template.pdb",
        target_seq="ABCDEFG",
    )

    # Verify the function called make_processed_dict_from_template
    mock_make_dict.assert_called_once_with(
        template_pdb="/test/path/ROCKET_inputs/template.pdb",
        target_seq="ABCDEFG",
        config_preset="model_1_ptm",
        device=device,
        msa_dict=None,
    )

    # Verify returned values
    assert result_dict == mock_processed_dict
    assert feature_key == "template_torsion_angles_sin_cos"
    assert torch.equal(features, mock_processed_dict["template_torsion_angles_sin_cos"])

    # Reset mocks for next test
    mock_make_dict.reset_mock()

    # Test with bias_version=1 (uses pickle loading)
    # Setup mock returns for glob, open, and pickle.load
    mock_glob.return_value = ["/test/path/predictions/test_processed_feats.pickle"]
    mock_pickle_dict = {"msa_feat": torch.ones(5, 10, 23), "aatype": torch.zeros(10)}
    mock_pickle_load.return_value = mock_pickle_dict

    # Call the function with bias_version=1
    result_dict, feature_key, features = init_processed_dict(
        bias_version=1, path="/test/path", device=device
    )

    # Verify the glob pattern was correct
    mock_glob.assert_called_once_with("/test/path/predictions/*processed_feats.pickle")

    # Verify pickle.load was called
    mock_pickle_load.assert_called_once()

    # Verify returned values
    assert feature_key == "msa_feat"
    assert torch.equal(features, mock_pickle_dict["msa_feat"])
    assert "msa_feat" in result_dict
    assert "aatype" in result_dict


def test_init_llgloss():
    """Test init_llgloss creates LLGloss object with correct parameters."""
    # Mock the structure factor object
    mock_sfc = MagicMock()
    mock_sfc.dHKL = torch.tensor([1.5, 2.0, 2.5, 3.0, 3.5])
    mock_sfc.device = "cpu"

    # Mock the tng_file path
    mock_tng_file = "/path/to/test.tng"

    # Mock the LLGloss constructor
    with patch("rocket.xtal.targets.LLGloss") as mock_llgloss_class:
        # Create a mock LLGloss object to be returned
        mock_llgloss = MagicMock()
        mock_llgloss_class.return_value = mock_llgloss

        # Call the function with default resolution parameters
        result = init_llgloss(mock_sfc, mock_tng_file)

        # Verify LLGloss was created with correct parameters
        mock_llgloss_class.assert_called_once_with(
            mock_sfc,
            mock_tng_file,
            mock_sfc.device,
            min(mock_sfc.dHKL),
            max(mock_sfc.dHKL),
        )

        # Verify the function returns the LLGloss object
        assert result == mock_llgloss

        # Reset mock for next test
        mock_llgloss_class.reset_mock()

        # Test with custom resolution parameters
        min_res = 2.0
        max_res = 3.0
        result = init_llgloss(
            mock_sfc, mock_tng_file, min_resolution=min_res, max_resolution=max_res
        )  # noqa: E501

        # Verify LLGloss was created with custom resolution parameters
        mock_llgloss_class.assert_called_once_with(
            mock_sfc, mock_tng_file, mock_sfc.device, min_res, max_res
        )


@patch("torch.optim.Adam")
@patch("torch.optim.AdamW")
def test_init_bias(mock_adamw, mock_adam):
    """Test init_bias function with different bias versions."""
    # Mock the device and learning rates
    device = torch.device("cpu")
    lr_a = 1e-3
    lr_m = 1e-4

    # Create a mock device_processed_features with minimum required data
    mock_features = {
        "aatype": torch.zeros(10)  # Simulate a sequence with 10 residues
    }

    # Set up the mocked optimizers
    mock_optimizer = MagicMock()
    mock_adam.return_value = mock_optimizer
    mock_adamw.return_value = mock_optimizer

    # Test bias_version=3
    result_features, optimizer, bias_names = init_bias(
        device_processed_features=mock_features,
        bias_version=3,
        device=device,
        lr_a=lr_a,
        lr_m=lr_m,
    )

    # Check that msa_feat_bias was created with correct shape
    assert "msa_feat_bias" in result_features
    assert result_features["msa_feat_bias"].shape == (512, 10, 23)
    assert result_features["msa_feat_weights"].shape == (512, 10, 23)
    assert result_features["msa_feat_bias"].requires_grad

    # Check optimizer was created with correct parameters
    mock_adam.assert_called_once()
    assert bias_names == ["msa_feat_bias", "msa_feat_weights"]

    # Reset mocks for next test
    mock_adam.reset_mock()
    mock_adamw.reset_mock()

    # Test bias_version=1
    result_features, optimizer, bias_names = init_bias(
        device_processed_features=mock_features,
        bias_version=1,
        device=device,
        lr_a=lr_a,
        lr_m=lr_m,
    )

    # Check that msa_feat_bias was created with correct shape
    assert "msa_feat_bias" in result_features
    assert result_features["msa_feat_bias"].shape == (512, 10, 23)
    assert result_features["msa_feat_bias"].requires_grad

    # Check optimizer was created with correct parameters
    mock_adam.assert_called_once()
    assert bias_names == ["msa_feat_bias"]

    # Reset mocks for next test
    mock_adam.reset_mock()
    mock_adamw.reset_mock()

    # Test bias_version=4
    result_features, optimizer, bias_names = init_bias(
        device_processed_features={
            "aatype": torch.zeros(10),
            "template_torsion_angles_sin_cos": torch.zeros(10, 5, 2),
        },
        bias_version=4,
        device=device,
        lr_a=lr_a,
        lr_m=lr_m,
    )

    # Check that template_torsion_angles_sin_cos_bias was created
    assert "template_torsion_angles_sin_cos_bias" in result_features
    assert result_features["template_torsion_angles_sin_cos_bias"].shape == (10, 5, 2)
    assert result_features["template_torsion_angles_sin_cos_bias"].requires_grad

    # Check bias names
    assert bias_names == ["template_torsion_angles_sin_cos_bias"]

    # Reset mocks for weight decay test
    mock_adam.reset_mock()
    mock_adamw.reset_mock()

    # Test with weight_decay
    result_features, optimizer, bias_names = init_bias(
        device_processed_features=mock_features,
        bias_version=2,
        device=device,
        lr_a=lr_a,
        lr_m=lr_m,
        weight_decay=0.01,
    )

    # Check AdamW was called instead of Adam
    mock_adamw.assert_called_once()
    mock_adam.assert_not_called()


@patch("rocket.coordinates.extract_allatoms")
@patch("rocket.coordinates.iterative_kabsch_alignment")
@patch("rocket.utils.plddt2pseudoB_pt")
@patch("rocket.utils.weighting")
@patch("rocket.utils.assert_numpy")
def test_position_alignment(
    mock_assert_numpy,
    mock_weighting,
    mock_plddt2pseudoB,
    mock_weighted_kabsch,
    mock_extract_allatoms,
):
    """Test position_alignment with mocked dependencies."""
    # Create mock inputs
    af2_output = {
        "plddt": torch.ones(10) * 90.0  # 10 residues with good plddt scores
    }
    device_processed_features = {}
    cra_name = ["A-1-GLY-CA", "A-1-GLY-N", "A-2-ALA-CA"]
    best_pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    exclude_res = []

    # Configure mocks
    # extract_allatoms returns coordinates and plddts
    mock_extract_allatoms.return_value = (
        torch.ones((3, 3)),  # xyz coordinates
        torch.ones(3),  # plddts for atoms
    )

    # Configure assert_numpy to return numpy arrays
    mock_assert_numpy.side_effect = lambda x: (
        x.numpy() if isinstance(x, torch.Tensor) else x
    )  # noqa: E501

    # Configure plddt2pseudoB to return B-factors
    pseudo_bs = torch.ones(3) * 30.0
    mock_plddt2pseudoB.return_value = pseudo_bs

    # Configure weighting to return weights
    weights = np.ones(3)
    mock_weighting.return_value = weights

    # Configure weighted_kabsch to return aligned coordinates
    aligned_xyz = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]])
    mock_weighted_kabsch.return_value = aligned_xyz

    # Call the function
    result_xyz, result_plddts, result_bs = position_alignment(
        af2_output, device_processed_features, cra_name, best_pos, exclude_res
    )

    # Verify the function works as expected
    assert torch.equal(result_xyz, aligned_xyz)
    assert torch.equal(torch.tensor(result_plddts), af2_output["plddt"])
    assert torch.equal(result_bs, pseudo_bs)

    # Check that extract_allatoms was called with correct arguments
    mock_extract_allatoms.assert_called_once_with(
        af2_output, device_processed_features, cra_name
    )

    # Check that weighted_kabsch was called with correct arguments
    mock_weighted_kabsch.assert_called_once()
    args, kwargs = mock_weighted_kabsch.call_args
    assert torch.equal(args[0], torch.ones((3, 3)))  # xyz_orth_sfc
    assert torch.equal(args[1], best_pos)  # best_pos
    assert args[2] == cra_name  # cra_name
    assert np.array_equal(kwargs["weights"], weights)  # weights
    assert kwargs["exclude_res"] == exclude_res  # exclude_res


@patch("rocket.coordinates.extract_allatoms")
@patch("rocket.coordinates.iterative_kabsch_alignment")
@patch("rocket.utils.plddt2pseudoB_pt")
@patch("rocket.utils.weighting")
@patch("rocket.utils.assert_numpy")
def test_position_alignment_with_reference_bfactor(
    mock_assert_numpy,
    mock_weighting,
    mock_plddt2pseudoB,
    mock_weighted_kabsch,
    mock_extract_allatoms,
):
    """Test position_alignment with reference_bfactor parameter."""
    # Create mock inputs
    af2_output = {"plddt": torch.ones(10) * 90.0}
    device_processed_features = {}
    cra_name = ["A-1-GLY-CA", "A-1-GLY-N", "A-2-ALA-CA"]
    best_pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    exclude_res = []
    reference_bfactor = torch.ones(3) * 20.0

    # Configure mocks as before
    mock_extract_allatoms.return_value = (torch.ones((3, 3)), torch.ones(3))
    mock_assert_numpy.side_effect = lambda x: (
        x.numpy() if isinstance(x, torch.Tensor) else x
    )  # noqa: E501
    mock_plddt2pseudoB.return_value = torch.ones(3) * 30.0
    mock_weighting.return_value = np.ones(3)
    mock_weighted_kabsch.return_value = torch.tensor([
        [1.5, 2.5, 3.5],
        [4.5, 5.5, 6.5],
        [7.5, 8.5, 9.5],
    ])

    # Call the function with reference_bfactor
    position_alignment(
        af2_output,
        device_processed_features,
        cra_name,
        best_pos,
        exclude_res,
        reference_bfactor=reference_bfactor,
    )

    # Check that weighting was called with reference_bfactor instead of pseudo_Bs
    mock_weighting.assert_called_once()
    args = mock_weighting.call_args[0]
    # Should be using reference_bfactor.numpy()
    assert np.array_equal(args[0], reference_bfactor.numpy())


def test_update_sigmaA():
    """Test update_sigmaA with mocked LLGloss objects."""
    # Create mock LLGloss objects
    llgloss = MagicMock()
    llgloss_rbr = MagicMock()

    # Create a mock for aligned_xyz
    aligned_xyz = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Set up return values for compute_Ecalc
    mock_ecalc = torch.tensor([0.5, 0.6, 0.7])
    mock_fc = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
    llgloss.compute_Ecalc.return_value = (mock_ecalc, mock_fc)

    mock_ecalc_rbr = torch.tensor([0.4, 0.5, 0.6])
    mock_fc_rbr = torch.tensor([0.5 + 1.0j, 1.5 + 2.0j, 2.5 + 3.0j])
    llgloss_rbr.compute_Ecalc.return_value = (mock_ecalc_rbr, mock_fc_rbr)

    # Test with no constant_fp parameters
    result_llgloss, result_llgloss_rbr, result_ecalc, result_fc = update_sigmaA(
        llgloss, llgloss_rbr, aligned_xyz
    )

    # Verify that compute_Ecalc was called correctly
    llgloss.compute_Ecalc.assert_called_once()
    # Use ANY for tensor arguments to avoid tensor comparison issues
    args, kwargs = llgloss.compute_Ecalc.call_args
    assert torch.equal(args[0], aligned_xyz.detach())
    assert kwargs["return_Fc"] is True
    assert kwargs["update_scales"] is True
    assert kwargs["added_chain_HKL"] is None
    assert kwargs["added_chain_asu"] is None

    # Verify that compute_Ecalc was called correctly for llgloss_rbr
    llgloss_rbr.compute_Ecalc.assert_called_once()
    args, kwargs = llgloss_rbr.compute_Ecalc.call_args
    assert torch.equal(args[0], aligned_xyz.detach())
    assert kwargs["return_Fc"] is True
    assert kwargs["solvent"] is False
    assert kwargs["update_scales"] is True
    assert kwargs["added_chain_HKL"] is None
    assert kwargs["added_chain_asu"] is None

    # Verify that refine_sigmaA_newton was called with correct parameters
    llgloss.refine_sigmaA_newton.assert_called_once()
    args, kwargs = llgloss.refine_sigmaA_newton.call_args
    assert torch.equal(args[0], mock_ecalc)
    assert kwargs["n_steps"] == 5
    assert kwargs["subset"] == "working"
    assert kwargs["smooth_overall_weight"] == 0.0

    # Verify that refine_sigmaA_newton was called correctly for llgloss_rbr
    llgloss_rbr.refine_sigmaA_newton.assert_called_once()
    args, kwargs = llgloss_rbr.refine_sigmaA_newton.call_args
    assert torch.equal(args[0], mock_ecalc_rbr)
    assert kwargs["n_steps"] == 2
    assert kwargs["subset"] == "working"
    assert kwargs["smooth_overall_weight"] == 0.0

    # Verify that the function returns the expected values
    assert result_llgloss == llgloss
    assert result_llgloss_rbr == llgloss_rbr
    assert torch.equal(result_ecalc, mock_ecalc)
    assert torch.equal(result_fc, mock_fc)

    # Reset mocks for the next test
    llgloss.reset_mock()
    llgloss_rbr.reset_mock()

    # Test with constant_fp parameters
    constant_fp_added_HKL = torch.tensor([0.1, 0.2, 0.3])
    constant_fp_added_asu = torch.tensor([0.4, 0.5, 0.6])

    update_sigmaA(
        llgloss,
        llgloss_rbr,
        aligned_xyz,
        constant_fp_added_HKL=constant_fp_added_HKL,
        constant_fp_added_asu=constant_fp_added_asu,
    )

    # Verify that compute_Ecalc was called with constant_fp parameters
    llgloss.compute_Ecalc.assert_called_once()
    args, kwargs = llgloss.compute_Ecalc.call_args
    assert torch.equal(args[0], aligned_xyz.detach())
    assert kwargs["return_Fc"] is True
    assert kwargs["update_scales"] is True
    assert torch.equal(kwargs["added_chain_HKL"], constant_fp_added_HKL)
    assert torch.equal(kwargs["added_chain_asu"], constant_fp_added_asu)

    # Verify that compute_Ecalc was called with constant_fp parameters for llgloss_rbr
    llgloss_rbr.compute_Ecalc.assert_called_once()
    args, kwargs = llgloss_rbr.compute_Ecalc.call_args
    assert torch.equal(args[0], aligned_xyz.detach())
    assert kwargs["return_Fc"] is True
    assert kwargs["solvent"] is False
    assert kwargs["update_scales"] is True
    assert torch.equal(kwargs["added_chain_HKL"], constant_fp_added_HKL)
    assert torch.equal(kwargs["added_chain_asu"], constant_fp_added_asu)


def tensor_call_equal(call1, call2):
    """Helper function to compare calls containing tensors."""
    if len(call1) != len(call2):
        return False

    for arg1, arg2 in zip(call1, call2, strict=False):
        if isinstance(arg1, torch.Tensor) and isinstance(arg2, torch.Tensor):
            if not torch.equal(arg1, arg2):
                return False
        elif arg1 != arg2:
            return False
    return True


@patch("rocket.xtal.utils.sigmaA_from_model")
def test_sigmaA_from_true(mock_sigmaA_from_model):
    """Test sigmaA_from_true with mocked dependencies."""
    # Create mock LLGloss objects
    llgloss = MagicMock()
    llgloss_rbr = MagicMock()

    # Configure bin_labels and dHKL properties that will be accessed
    llgloss.bin_labels = torch.tensor([1, 2, 3])
    llgloss.sfc.dHKL = torch.tensor([2.0, 2.5, 3.0])

    # Create mock aligned_xyz, Etrue, and phitrue
    aligned_xyz = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    Etrue = torch.tensor([1.1, 1.2, 1.3])
    phitrue = torch.tensor([0.1, 0.2, 0.3])

    # Set up return values for compute_Ecalc
    mock_ecalc = torch.tensor([0.5, 0.6, 0.7])
    mock_fc = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
    llgloss.compute_Ecalc.return_value = (mock_ecalc, mock_fc)

    mock_ecalc_rbr = torch.tensor([0.4, 0.5, 0.6])
    mock_fc_rbr = torch.tensor([0.5 + 1.0j, 1.5 + 2.0j, 2.5 + 3.0j])
    llgloss_rbr.compute_Ecalc.return_value = (mock_ecalc_rbr, mock_fc_rbr)

    # Mock sigmaA_from_model return values
    mock_sigmas = torch.tensor([0.8, 0.7, 0.6])
    mock_sigmas_rbr = torch.tensor([0.85, 0.75, 0.65])
    mock_sigmaA_from_model.side_effect = [mock_sigmas, mock_sigmas_rbr]

    # Call the function
    result_llgloss, result_llgloss_rbr = sigmaA_from_true(
        llgloss, llgloss_rbr, aligned_xyz, Etrue, phitrue
    )

    # Verify compute_Ecalc was called correctly
    llgloss.compute_Ecalc.assert_called_once()
    args, kwargs = llgloss.compute_Ecalc.call_args
    assert torch.equal(args[0], aligned_xyz.detach())
    assert kwargs["return_Fc"] is True
    assert kwargs["update_scales"] is True
    assert kwargs["added_chain_HKL"] is None
    assert kwargs["added_chain_asu"] is None

    # Verify compute_Ecalc was called correctly for llgloss_rbr
    llgloss_rbr.compute_Ecalc.assert_called_once()
    args, kwargs = llgloss_rbr.compute_Ecalc.call_args
    assert torch.equal(args[0], aligned_xyz.detach())
    assert kwargs["return_Fc"] is True
    assert kwargs["solvent"] is False
    assert kwargs["update_scales"] is True
    assert kwargs["added_chain_HKL"] is None
    assert kwargs["added_chain_asu"] is None

    # Check if sigmaA_from_model was called with the expected arguments
    assert mock_sigmaA_from_model.call_count == 2

    # Check first call arguments (for llgloss)
    args_llgloss = mock_sigmaA_from_model.call_args_list[0][0]
    assert torch.equal(args_llgloss[0], Etrue)
    assert torch.equal(args_llgloss[1], phitrue)
    assert torch.equal(args_llgloss[2], mock_ecalc)
    assert torch.equal(args_llgloss[3], mock_fc)
    assert torch.equal(args_llgloss[4], llgloss.sfc.dHKL)
    assert torch.equal(args_llgloss[5], llgloss.bin_labels)

    # Check second call arguments (for llgloss_rbr)
    args_llgloss_rbr = mock_sigmaA_from_model.call_args_list[1][0]
    assert torch.equal(args_llgloss_rbr[0], Etrue)
    assert torch.equal(args_llgloss_rbr[1], phitrue)
    assert torch.equal(args_llgloss_rbr[2], mock_ecalc_rbr)
    assert torch.equal(args_llgloss_rbr[3], mock_fc_rbr)
    assert torch.equal(args_llgloss_rbr[4], llgloss.sfc.dHKL)
    assert torch.equal(args_llgloss_rbr[5], llgloss.bin_labels)

    # Verify sigmaAs properties were updated correctly
    assert torch.equal(llgloss.sigmaAs, mock_sigmas)
    assert torch.equal(llgloss_rbr.sigmaAs, mock_sigmas_rbr)

    # Verify the function returns the modified llgloss objects
    assert result_llgloss is llgloss
    assert result_llgloss_rbr is llgloss_rbr


@patch("rocket.xtal.utils.sigmaA_from_model")
def test_sigmaA_from_true_with_constant_fp(mock_sigmaA_from_model):
    """Test sigmaA_from_true with constant_fp parameters."""
    # Create mock LLGloss objects
    llgloss = MagicMock()
    llgloss_rbr = MagicMock()

    # Configure necessary properties
    llgloss.bin_labels = torch.tensor([1, 2, 3])
    llgloss.sfc.dHKL = torch.tensor([2.0, 2.5, 3.0])

    # Mock input parameters
    aligned_xyz = torch.tensor([[1.0, 2.0, 3.0]])
    Etrue = torch.tensor([1.1, 1.2, 1.3])
    phitrue = torch.tensor([0.1, 0.2, 0.3])
    constant_fp_added_HKL = torch.tensor([0.1, 0.2, 0.3])
    constant_fp_added_asu = torch.tensor([0.4, 0.5, 0.6])

    # Configure compute_Ecalc return values
    llgloss.compute_Ecalc.return_value = (
        torch.tensor([0.5]),
        torch.tensor([1.0 + 2.0j]),
    )
    llgloss_rbr.compute_Ecalc.return_value = (
        torch.tensor([0.4]),
        torch.tensor([0.5 + 1.0j]),
    )  # noqa: E501

    # Configure sigmaA_from_model return values
    mock_sigmaA_from_model.side_effect = [torch.tensor([0.8]), torch.tensor([0.85])]

    # Call the function with constant_fp parameters
    sigmaA_from_true(
        llgloss,
        llgloss_rbr,
        aligned_xyz,
        Etrue,
        phitrue,
        constant_fp_added_HKL=constant_fp_added_HKL,
        constant_fp_added_asu=constant_fp_added_asu,
    )

    # Verify compute_Ecalc was called with constant_fp parameters
    args, kwargs = llgloss.compute_Ecalc.call_args
    assert torch.equal(kwargs["added_chain_HKL"], constant_fp_added_HKL)
    assert torch.equal(kwargs["added_chain_asu"], constant_fp_added_asu)

    # Verify compute_Ecalc was called with constant_fp parameters for llgloss_rbr
    args, kwargs = llgloss_rbr.compute_Ecalc.call_args
    assert torch.equal(kwargs["added_chain_HKL"], constant_fp_added_HKL)
    assert torch.equal(kwargs["added_chain_asu"], constant_fp_added_asu)
