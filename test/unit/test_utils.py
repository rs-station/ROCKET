# Mock external dependencies before importing utils
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from rocket.utils import (
    apply_resolution_cutoff,
    assert_numpy,
    assert_tensor,
    convert_feat_tensors_to_numpy,
    d2q,
    dict_map,
    get_params_path,
    load_mtz,
    load_pdb,
    move_tensors_to_device,
    move_tensors_to_device_inplace,
    plddt2pseudoB_np,
    plddt2pseudoB_pt,
    tree_map,
    try_gpu,
    weighting,
    weighting_torch,
)


def test_plddt2pseudoB_basic():
    arr = np.array([0, 50, 100])
    out = plddt2pseudoB_np(arr)
    assert out.shape == arr.shape
    assert np.all(out > 0)
    assert np.isfinite(out).all()


def test_plddt2pseudoB_empty():
    arr = np.array([])
    out = plddt2pseudoB_np(arr)
    assert out.shape == arr.shape


def test_plddt2pseudoB_known_values():
    out = plddt2pseudoB_np(np.array([70.0, 100.0, 0.0]))
    # At pLDDT=70: delta = 1.5 -> B = 8*pi^2/3 * 1.5^2 = 6*pi^2
    np.testing.assert_allclose(out[0], 6 * np.pi**2, rtol=1e-10)
    np.testing.assert_allclose(
        out[1], 8 * np.pi**2 / 3 * (1.5 * np.exp(-1.2)) ** 2, rtol=1e-10
    )
    np.testing.assert_allclose(
        out[2], 8 * np.pi**2 / 3 * (1.5 * np.exp(2.8)) ** 2, rtol=1e-10
    )


def test_plddt2pseudoB_monotonic():
    plddts = np.linspace(0, 100, 50)
    out = plddt2pseudoB_np(plddts)
    assert np.all(np.diff(out) < 0)


def test_plddt2pseudoB_np_pt_parity():
    plddts = np.linspace(0, 100, 25)
    np_out = plddt2pseudoB_np(plddts)
    pt_out = plddt2pseudoB_pt(torch.tensor(plddts, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(np_out, pt_out, rtol=1e-10)


def test_plddt2pseudoB_pt_shape_and_dtype():
    x = torch.tensor([10.0, 50.0, 90.0])
    out = plddt2pseudoB_pt(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.all(out > 0)
    assert torch.isfinite(out).all()


def test_plddt2pseudoB_pt_autograd():
    x = torch.tensor([30.0, 70.0, 95.0], requires_grad=True)
    out = plddt2pseudoB_pt(x)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    # dB/d(pLDDT) is negative everywhere (B strictly decreases with pLDDT)
    assert torch.all(x.grad < 0)


def test_plddt2pseudoB_pt_empty():
    out = plddt2pseudoB_pt(torch.tensor([]))
    assert out.shape == (0,)


def test_weighting_cutoffs():
    b = np.array([10.0, 11.5, 20.0, 30.0, 40.0])
    w = weighting(b)
    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[1], 1.0)
    assert 0.5 < w[2] < 1.0
    assert np.isclose(w[3], 0.5)
    assert w[4] < 0.5


def test_weighting_torch_matches_numpy():
    b = torch.tensor([10.0, 11.5, 20.0, 30.0, 40.0])
    wn = weighting(b.numpy())
    wt = weighting_torch(b).numpy()
    np.testing.assert_allclose(wn, wt, rtol=1e-5)


def test_weighting_scalar():
    w = weighting(10.0)
    assert w == 1.0


def test_dict_map_exclude_keys():
    d = {"a": 1, "msa_feat_bias": 2}

    def fn(x):
        return x + 1

    out = dict_map(fn, d, int)
    assert out["a"] == 2
    assert out["msa_feat_bias"] == 2  # should not be mapped


def test_tree_map_nested():
    tree = {"a": [1, 2], "b": (3, 4)}
    out = tree_map(lambda x: x * 2, tree, int)
    assert out["a"] == [2, 4]
    assert out["b"] == (6, 8)


def test_tree_map_non_supported():
    class Dummy:
        pass

    with pytest.raises(ValueError):
        tree_map(lambda x: x, Dummy(), int)


def test_try_gpu_cpu():
    with patch("torch.cuda.device_count", return_value=0):
        dev = try_gpu()
        assert str(dev) == "cpu"


def test_try_gpu_cuda():
    with patch("torch.cuda.device_count", return_value=2):
        dev = try_gpu(1)
        assert str(dev) == "cuda:1"


def test_move_tensors_to_device():
    d = {"a": torch.zeros(2), "b": 3}
    out = move_tensors_to_device(d, device="cpu")
    assert out["a"].device.type == "cpu"
    assert out["b"] == 3


def test_move_tensors_to_device_inplace():
    d = {"a": torch.zeros(2)}
    move_tensors_to_device_inplace(d, device="cpu")
    assert d["a"].device.type == "cpu"


def test_convert_feat_tensors_to_numpy():
    d = {"a": torch.ones(2), "b": "notensor"}
    out = convert_feat_tensors_to_numpy(d)
    assert isinstance(out["a"], np.ndarray)
    assert out["b"] == "notensor"


def test_assert_numpy_from_tensor():
    arr = torch.ones(2)
    np_arr = assert_numpy(arr)
    assert isinstance(np_arr, np.ndarray)


def test_assert_numpy_from_list():
    lst = [1, 2, 3]
    np_arr = assert_numpy(lst)
    assert isinstance(np_arr, np.ndarray)


def test_assert_tensor_from_np():
    arr = np.ones(2)
    tensor = assert_tensor(arr, device="cpu")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type == "cpu"


def test_assert_tensor_from_list():
    lst = [1, 2, 3]
    tensor = assert_tensor(lst, device="cpu")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type == "cpu"


def test_d2q_scalar_and_array():
    assert np.isclose(d2q(2.0), np.pi)
    arr = np.array([1.0, 2.0])
    out = d2q(arr)
    assert np.allclose(out, 2 * np.pi / arr)


def test_load_mtz():
    with patch("rocket.utils.rs.read_mtz") as mock_read:
        mock_ds = MagicMock()
        mock_read.return_value = mock_ds
        load_mtz("file.mtz")
        mock_read.assert_called_once_with("file.mtz")
        mock_ds.compute_dHKL.assert_called_once_with(inplace=True)


def test_load_pdb():
    with patch("rocket.utils.PDBParser") as mock_pdb:
        load_pdb("file.pdb")
        mock_pdb.assert_called_once_with("file.pdb")


def test_get_params_path(monkeypatch):
    monkeypatch.setenv("OPENFOLD_RESOURCES", "/tmp/resources")
    path = get_params_path()
    assert path == "/tmp/resources/params"


def test_get_params_path_missing(monkeypatch):
    monkeypatch.delenv("OPENFOLD_RESOURCES", raising=False)
    with pytest.raises(ValueError):
        get_params_path()


def test_apply_resolution_cutoff_with_dataset():
    """
    Test apply_resolution_cutoff with a DataSet object and explicit max_resolution.
    """

    # Setup mock dataset
    mock_ds = MagicMock()
    mock_ds.dHKL = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Mock the filtering and copy chain
    filtered_mock = MagicMock()
    mock_ds.__getitem__.return_value = filtered_mock

    # Call the function
    result = apply_resolution_cutoff(mock_ds, min_resolution=2.0, max_resolution=4.0)

    # Assertions
    mock_ds.compute_dHKL.assert_called_once_with(inplace=True)

    # Check that the filtering happened correctly
    args, _ = mock_ds.__getitem__.call_args
    mask = args[0]
    expected_mask = (mock_ds.dHKL >= 2.0) & (mock_ds.dHKL <= 4.0)
    np.testing.assert_array_equal(mask, expected_mask)

    # Check that copy() was called on the filtered result
    filtered_mock.copy.assert_called_once()

    # Check that the final result is the copied, filtered mock
    assert result == filtered_mock.copy.return_value


def test_apply_resolution_cutoff_with_filepath():
    """Test apply_resolution_cutoff with a file path string."""

    with patch("rocket.utils.rs.read_mtz") as mock_read_mtz:
        # Setup mock dataset
        mock_ds = MagicMock()
        mock_ds.dHKL = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_read_mtz.return_value = mock_ds

        # Mock the filtering and copy chain
        filtered_mock = MagicMock()
        mock_ds.__getitem__.return_value = filtered_mock

        # Call the function
        apply_resolution_cutoff("test.mtz", min_resolution=2.0, max_resolution=4.0)

        # Assertions
        mock_read_mtz.assert_called_once_with("test.mtz")
        mock_ds.compute_dHKL.assert_called_once_with(inplace=True)

        # Check filtering
        args, _ = mock_ds.__getitem__.call_args
        mask = args[0]
        expected_mask = (mock_ds.dHKL >= 2.0) & (mock_ds.dHKL <= 4.0)
        np.testing.assert_array_equal(mask, expected_mask)

        filtered_mock.copy.assert_called_once()


def test_apply_resolution_cutoff_no_max_resolution():
    """Test apply_resolution_cutoff when max_resolution is not provided."""

    # Setup mock dataset
    mock_ds = MagicMock()
    mock_ds.dHKL = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # .max() will be 5.0

    # Mock the filtering and copy chain
    filtered_mock = MagicMock()
    mock_ds.__getitem__.return_value = filtered_mock

    # Call the function
    apply_resolution_cutoff(mock_ds, min_resolution=3.0)

    # Assertions
    mock_ds.compute_dHKL.assert_called_once_with(inplace=True)

    # Check filtering logic
    args, _ = mock_ds.__getitem__.call_args
    mask = args[0]
    # max_resolution should be inferred as 5.0
    expected_mask = (mock_ds.dHKL >= 3.0) & (mock_ds.dHKL <= 5.0)
    np.testing.assert_array_equal(mask, expected_mask)

    filtered_mock.copy.assert_called_once()
