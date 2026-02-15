# ROCKET YAML parameters (minimal)

This page lists the minimum fields needed for a successful run and the most important optional settings.

## Required for all runs

- `paths.input_pdb`: Path to the starting model PDB.
- `paths.target_map`: Path to the target CCP4/MAP file.
- `paths.input_dir`: Base input directory (used to resolve relative paths).
- `paths.file_id`: Identifier used for outputs and fallbacks.
- `data.min_resolution`: Minimum resolution (required for structure factor calculation).
- `execution.cuda_device`: GPU index to use.
- `panddamap.output_dir`: Output directory for refinement.

## Required to build features

You must provide **one** of the following:

- `paths.alignment_dir` with `.a3m` or `.sto` files, **or**
- `paths.msa_feat_init_path` (processed features pickle), **or**
- `paths.input_fasta` (to query MMseqs2 server and generate MSAs [TODO!] ).

If none are available, the run will stop with an error.

## Pandda map masking vs gradient masking

There are two distinct masks that affect refinement:

### 1) Loss mask (map comparison)
- `panddamap.ligand_centroid`: Center of the spherical mask (list of three floats).
- `panddamap.pandda_map_radius`: Radius in Angstroms (default 15.0).

This mask limits **which voxels** in the target and model maps are compared
for the real-space loss. It does not directly change the structure factors
or which atoms contribute to the model map.

If `panddamap.ligand_centroid` is null, the loss mask is disabled and the
loss compares the full map.

### 2) Gradient mask (residue-level)
The same `ligand_centroid` and `pandda_map_radius` are also used to restrict
**which residues receive gradients** during map-based refinement:

- Any residue with **at least one atom** inside the spherical radius is
	allowed to receive gradients.
- All other residues are detached from the computational graph.

This prevents gradients from propagating to regions far from the ligand site.

## Preprocessing the target map

- `panddamap.preprocess_target_map`: Enables ligand-centered masking and optional denoise.
- `paths.ligand_pdb`: Required if preprocessing is enabled.
- `panddamap.ligand_mask_radius`: Radius for ligand mask in preprocessing.
- `panddamap.tv_denoise`: Enable TV denoising (optional).
- `panddamap.denoise_high_res_limit`: High-resolution limit for denoise.

## MMseqs2 (server MSA) [TODO!]

Used only when alignments and processed features are missing and `paths.input_fasta` is present.

- `paths.mmseqs2_host_url`: Default `https://api.colabfold.com`.
- `paths.mmseqs2_user_agent`: Identify your client (required for good API etiquette).
- `paths.mmseqs2_use_env`: Use environmental databases.
- `paths.mmseqs2_use_filter`: Apply diversity filtering.
- `paths.mmseqs2_output_dir`: Directory for downloaded alignments.

## MSE prepass (optional)

- `panddamap.mse_prepass.enabled`: Run an MSE refinement before real-space.
- `panddamap.mse_prepass.selection`: `ALL`, `CA`, or `BB` (default `BB`).
- `panddamap.mse_prepass.save_biases`: Save MSE bias weights for reuse.
- `panddamap.mse_prepass.first_n_residues`: Optional residue cap for MSE (debug).

## W&B (optional)

- `panddamap.use_wandb`: Enable logging.
- `panddamap.wandb_entity`, `panddamap.wandb_project`, `panddamap.wandb_name`.
- `panddamap.wandb_tags`: Tags; `mse` and `realspace` are appended automatically.

## Run controls

- `algorithm.use_deepspeed_evo_attention`: Enable DeepSpeed evo attention kernels (default true).

- `algorithm.iterations`: Number of iterations per run.
- `execution.num_of_runs`: Number of runs.
- `panddamap.save_best_pdb`, `panddamap.save_trajectory_pdb`.

## Common pitfalls

- `paths.ligand_pdb` is required if `panddamap.preprocess_target_map` is true.
- `data.min_resolution` is required.
