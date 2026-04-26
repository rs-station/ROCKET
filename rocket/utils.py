import os
from functools import partial

import numpy as np
import reciprocalspaceship as rs
import torch
from scipy.interpolate import RegularGridInterpolator
from SFC_Torch import PDBParser
from SFC_Torch.mask import reciprocal_grid


def plddt2pseudoB_np(plddts):
    # Use Tom Terwilliger's formula to convert plddt to Bfactor
    deltas = 1.5 * np.exp(4 * (0.7 - 0.01 * plddts))
    b_factors = (8 * np.pi**2 * deltas**2) / 3
    return b_factors


def plddt2pseudoB_pt(plddts):
    # Use Terwilliger's formula to convert plddt to pseudoB
    deltas = 1.5 * torch.exp(4 * (0.7 - 0.01 * plddts))
    b_factors = (8 * torch.pi**2 * deltas**2) / 3

    return b_factors


def cc_to_bfactor( 
    cc_values: torch.Tensor, 
    dmin: float, 
) -> torch.Tensor: 
    """Convert correlation coefficients to pseudo B-factors. 

    Args: 
        cc_values: Correlation coefficient values [N] 
        dmin: Minimum resolution in Angstroms 

    Returns: 
        Pseudo B-factors [N] 
    """ 
    # Empirical conversion: B = -8π²d²ln(CC) 
    cc_clamped = torch.clamp(cc_values, min=0.001, max=0.999) 
    b_factors = -8.0 * (torch.pi**2) * (dmin**2) * torch.log(cc_clamped) 
    return torch.clamp(b_factors, min=10.0, max=200.0) 


def weighting(x, cutoff1=11.5, cutoff2=30.0):
    """
    Convert B factor to weights for L2 loss and Kabsch Alignment
    w(B) =
    1.0                                           B <= cutoff1
    1.0 - 0.5*(B-cutoff1)/(cutoff2-cutoff1)       cutoff1 < B <= cutoff2
    0.5 * exp(-sqrt(B-cutoff2))                   cutoff2 < B
    """
    a = np.where(x <= cutoff1, 1.0, 1.0 - 0.5 * (x - cutoff1) / (cutoff2 - cutoff1))
    b = 0.5 * np.exp(-np.sqrt(np.where(x <= cutoff2, 1.0, x - cutoff2)))
    return np.where(a >= 0.5, a, b)


def weighting_torch(x, cutoff1=11.5, cutoff2=30.0):
    """
    pytorch implementation of the weighting function:
    w(B) =
    1.0                                           B <= cutoff1
    1.0 - 0.5*(B-cutoff1)/(cutoff2-cutoff1)       cutoff1 < B <= cutoff2
    0.5 * exp(-sqrt(B-cutoff2))                   cutoff2 < B
    """
    a = torch.where(x <= cutoff1, 1.0, 1.0 - 0.5 * (x - cutoff1) / (cutoff2 - cutoff1))
    b = 0.5 * torch.exp(-torch.sqrt(torch.where(x <= cutoff2, 1.0, x - cutoff2)))
    return torch.where(a >= 0.5, a, b)


def dict_map(fn, dic, leaf_type, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = {"msa_feat_bias", "msa_feat_weights"}

    new_dict = {}
    for k, v in dic.items():
        if k in exclude_keys:
            new_dict[k] = v  # Keep the key-value pair as it is
        elif isinstance(v, dict):
            new_dict[k] = dict_map(fn, v, leaf_type, exclude_keys)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        raise ValueError("Not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def move_tensors_to_device_inplace(processed_features, device=None):
    """
    Moves PyTorch tensors in a dictionary to the specified device in-place.

    Args:
        processed_features (dict): Dictionary containing tensors.
        device (str): Device to move tensors to (e.g., "cuda:0", "cpu").
    """
    if device is None:
        device = try_gpu()
    # Iterate through the keys and values in the input dictionary
    for key, value in processed_features.items():
        # Check if the value is a PyTorch tensor
        if isinstance(value, torch.Tensor):
            # Move the tensor to the specified device in-place
            processed_features[key] = value.to(device)


def move_tensors_to_device(processed_features, device=None):
    """
    Moves PyTorch tensors in a dictionary to the specified device.

    Args:
        processed_features (dict): Dictionary containing tensors.
        device (str): Device to move tensors to (e.g., "cuda:0", "cpu").

    Returns:
        dict: Dictionary with tensors moved to the specified device.
    """
    if device is None:
        device = try_gpu()
    processed_features_on_device = {}

    for key, value in processed_features.items():
        if isinstance(value, torch.Tensor):
            device_value = value.clone().to(device)
        else:
            device_value = value
        processed_features_on_device[key] = device_value

    return processed_features_on_device


def convert_feat_tensors_to_numpy(dictionary):
    numpy_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            numpy_dict[key] = value.detach().cpu().numpy()
        else:
            numpy_dict[key] = value
    return numpy_dict


def is_list_or_tuple(x):
    return isinstance(x, list | tuple)


def assert_numpy(x, arr_type=None):
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().numpy()
    if is_list_or_tuple(x):
        x = np.array(x)
    assert isinstance(x, np.ndarray)
    if arr_type is not None:
        x = x.astype(arr_type)
    return x


def assert_tensor(x, arr_type=None, device="cuda:0"):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, device=device)
    if is_list_or_tuple(x):
        x = np.array(x)
        x = torch.tensor(x, device=device)
    assert isinstance(x, torch.Tensor)
    if arr_type is not None:
        x = x.to(arr_type)
    return x


def d2q(d):
    return 2 * np.pi / d


def load_mtz(mtz: str) -> rs.DataSet:
    dataset = rs.read_mtz(mtz)
    dataset.compute_dHKL(inplace=True)
    return dataset


def load_pdb(pdb: str) -> PDBParser:
    model = PDBParser(pdb)
    return model


def get_params_path():
    resources_path = os.environ.get("OPENFOLD_RESOURCES", None)
    if resources_path is None:
        raise ValueError("Please set OPENFOLD_RESOURCES environment variable")
    params_path = os.path.join(resources_path, "params")
    return params_path


def apply_resolution_cutoff(
    dataset: rs.DataSet | str,
    min_resolution: float = None,
    max_resolution: float = None,
):
    """
    Apply resolution cutoff to a dataset.

    Args:
        dataset (rs.DataSet | str): The dataset to filter, can be a path to an MTZ file
            or an rs.DataSet object.
        min_resolution (float): Minimum resolution to keep.
        max_resolution (float): Maximum resolution to keep.

    Returns:
        rs.DataSet: Filtered dataset with applied resolution cutoff.
    """
    if isinstance(dataset, str):
        dataset = rs.read_mtz(dataset)

    dataset.compute_dHKL(inplace=True)
    if max_resolution is None:
        max_resolution = dataset.dHKL.max()
    if min_resolution is None:
        min_resolution = dataset.dHKL.min()
    filtered_dataset = dataset[
        (dataset.dHKL >= (min_resolution - 1e-4))
        & (dataset.dHKL <= (max_resolution + 1e-4))
    ].copy()

    return filtered_dataset


def g_function_np(R: float, s: np.ndarray) -> np.ndarray:
    """
    Calculates the Fourier transform of a sphere (g-function) using
    spherical Bessel function in NumPy.

    Args:
        R (float): The radius of the sphere.
        s (np.ndarray): An array of reciprocal space vector magnitudes (1/d).

    Returns:
        np.ndarray: The calculated g-function values.
    """
    # Ensure s is a numpy array
    if not isinstance(s, np.ndarray):
        s = np.array(s, dtype=np.float64)

    # Create an output array initialized with the value for s=0
    g_vals = np.ones_like(s)

    # Identify non-zero s values to avoid division by zero
    nonzero_mask = np.abs(s) > 1e-9

    # Calculate the g-function only for non-zero s
    s_nonzero = s[nonzero_mask]
    x = 2 * np.pi * R * s_nonzero
    g_vals_nonzero = 3 * (np.sin(x) - x * np.cos(x)) / np.power(x, 3)

    # Place the calculated values into the output array
    g_vals[nonzero_mask] = g_vals_nonzero

    return g_vals


def map2fourier(map_grid, HKL_arrays):
    rs_grid = torch.fft.ifftn(map_grid, dim=(-3, -2, -1), norm="forward")
    tuple_index = tuple(torch.tensor(HKL_arrays.T, device=rs_grid.device, dtype=int))
    Frs = rs_grid[tuple_index]
    return Frs


def fourier2map(Fs, HKL_arrays, gridsize):
    rs_grid = reciprocal_grid(HKL_arrays, Fs, gridsize)
    map_grid = torch.real(torch.fft.fftn(rs_grid, dim=(-3, -2, -1)))
    return map_grid


def interpolate_grid_points(map_grid, frac_coords_batch, method="linear"):
    """
    Interpolates values on a 3D grid at a batch of fractional coordinates.

    Args:
        map_grid (np.ndarray): A 3D NumPy array representing the data grid.
        frac_coords_batch (np.ndarray): A 2D NumPy array of shape (N, 3),
                                        where N is the number of points and
                                        each row contains the [x, y, z]
                                        fractional coordinates.

    Returns:
        np.ndarray: A 1D NumPy array of shape (N,) containing the
                    interpolated values for each point in the batch.
    """
    # 1. Define the grid points (indices) for each dimension.
    x = np.arange(map_grid.shape[0])
    y = np.arange(map_grid.shape[1])
    z = np.arange(map_grid.shape[2])

    # 2. Create the interpolator object.
    interpolator = RegularGridInterpolator(
        (x, y, z), map_grid, method=method, bounds_error=False, fill_value=None
    )

    # 3. Convert the batch of fractional coordinates to grid coordinates.
    grid_shape = np.array(map_grid.shape)
    frac_coords_batch = frac_coords_batch % 1.0
    point_coords_batch = frac_coords_batch * (grid_shape - 1)

    # 4. Get the interpolated values for the entire batch of points.
    interpolated_values = interpolator(point_coords_batch)

    return interpolated_values


def get_rscc_from_Fmap(
    Fcalc: torch.Tensor,
    Fmap: torch.Tensor,
    HKL_arrays: np.ndarray,
    gridsize: list[int],
    Rg: torch.Tensor,
    unitcell_volume: float,
) -> float:
    assert len(Fcalc) == len(HKL_arrays)
    assert len(Fmap) == len(HKL_arrays)
    assert len(Rg) == len(HKL_arrays)

    map1 = fourier2map(Fcalc, HKL_arrays, gridsize)
    map1_normed = (map1 - map1.mean()) / map1.std()

    map2 = fourier2map(Fmap, HKL_arrays, gridsize)
    map2_normed = (map2 - map2.mean()) / map2.std()

    productmap = map1_normed * map2_normed
    Fproductmap = map2fourier(productmap, HKL_arrays)
    Fproductmap_smooth = Fproductmap * Rg

    productmap_smooth = fourier2map(Fproductmap_smooth, HKL_arrays, gridsize)
    productmap_smooth_norm = 2.0 * productmap_smooth / unitcell_volume

    variance1_map = map1_normed * map1_normed
    Fvmap1 = map2fourier(variance1_map, HKL_arrays)
    Fvmap1_smoothed = Fvmap1 * Rg
    vmap1_smoothed = fourier2map(Fvmap1_smoothed, HKL_arrays, gridsize)
    vmap1_smoothed_normed = 2.0 * vmap1_smoothed / unitcell_volume
    vmap1_smoothed_normed_offset = (
        vmap1_smoothed_normed - vmap1_smoothed_normed.min() + 1e-3
    )

    variance2_map = map2_normed * map2_normed
    Fvmap2 = map2fourier(variance2_map, HKL_arrays)
    Fvmap2_smoothed = Fvmap2 * Rg
    vmap2_smoothed = fourier2map(Fvmap2_smoothed, HKL_arrays, gridsize)
    vmap2_smoothed_normed = 2.0 * vmap2_smoothed / unitcell_volume
    vmap2_smoothed_normed_offset = (
        vmap2_smoothed_normed - vmap2_smoothed_normed.min() + 1e-3
    )

    ccmap = (productmap_smooth_norm - productmap_smooth_norm.mean()) / torch.sqrt(
        vmap1_smoothed_normed_offset * vmap2_smoothed_normed_offset
    )
    return ccmap.cpu().numpy()


def get_b_from_CC(CC_values, d_min):
    # Truncate very small or negative CC values
    CC_trunc = (CC_values + 0.001 + np.abs(CC_values - 0.001)) / 2

    # Flex arrays can't be used as exponents, so work around this with logs
    # TODO: verify the constants here
    # Randy's note: In case you’re wondering, the magic numbers and the magic formula
    # come from some things I did in Mathematica.
    # I worked out a formula that converts RMSD to an expected CC.
    u1 = np.exp(np.log(0.188779) * CC_trunc)
    u2 = np.exp(2.21742 * CC_trunc * np.log(CC_trunc))
    u3 = np.exp(0.777046 * np.tan(CC_trunc) * np.log(CC_trunc) * CC_trunc)
    b_values_from_CC = (
        np.power(d_min, 2) * 134.024 * np.power(0.213624 - u1 * u2 - 0.359452 * u3, 2)
    )
    # Previous Approach, now we dropped
    # Truncate B-values at a maximum of 350 above minimum
    # 350 was chosen semi-arbitrarily as a point at which the sigmaA curve drops
    # below 0.1 at 6A, so that data beyond 6A will have little influence
    # bmax = np.min(b_values_from_CC) + 350.0
    # b_values_from_CC = (b_values_from_CC + bmax - np.abs(b_values_from_CC - bmax)) / 2

    # Do normal truncation instead, as Randy Suggested
    bmin = np.min(b_values_from_CC)
    b_values_from_CC = b_values_from_CC - bmin
    bmax = 999.0
    b_values_from_CC = np.clip(b_values_from_CC, 0.0, bmax)
    return b_values_from_CC
