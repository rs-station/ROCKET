# ROCKET pipelines: rk.config and rk.refine

This page describes the two CLI entrypoints in this branch and how they work together.

## rk.config (config generator)

Purpose: create a starter YAML with reasonable defaults and auto-discovered paths.

What it does:
- Reads the input directory and attempts to locate:
  - `input_pdb`
  - `target_map`
  - `input_fasta`
  - `alignment_dir`
  - `msa_feat_init_path`
- Writes a `ROCKET_config.yaml` in the requested working directory.

Notes:
- It does not run any refinement.
- If it cannot find a file, the corresponding YAML field is left empty.

## rk.refine (PanDDA Map refinement)

Purpose: run a full refinement pipeline from a YAML config.

High-level steps:
1. Load the YAML config.
2. Optionally preprocess the target map (ligand mask/denoise).
3. Optional MSE prepass (if enabled):
   - Refines MSA biases using MSE loss to a known reference state (most likely the unbound state you have)
   - Outputs updated bias/weights for the real-space run follow-up.
4. Real-space refinement:
   - Builds features from alignment or processed features.
   - If no alignments or pickle are present and `input_fasta` is provided, it queries the MMseqs2 server to generate MSAs.
   - Runs real-space loss on the target map.

### Reusing biases from a previous run

If you want to start from previously optimized bias tensors, set these paths
in your YAML under `paths`:

- `starting_bias`: path to a saved `msa_feat_bias` tensor (.pt)
- `starting_weights`: path to a saved `msa_feat_weights` tensor (.pt)

These will be loaded and used to initialize the optimizer state in the next run.
You can point them at artifacts from a prior refinement output directory (e.g.
`best_msa_bias.pt` and `best_feat_weights.pt`). For example, if you already have a reference unbound state refined from a previous MSE Prepass, anytime you have a dataset from that target you can load them in and skip another prepass.

## Add-ons

### W&B logging
Enable with:
- `panddamap.use_wandb: true`
- `panddamap.wandb_entity`, `panddamap.wandb_project`, `panddamap.wandb_name`

The pipeline automatically appends `mse` and `realspace` tags based on the run type.

### MMseqs2 server [TODO]
Used only when both alignments and processed features are missing and `paths.input_fasta` is present.
Set these in `paths`:
- `mmseqs2_host_url`
- `mmseqs2_user_agent`
- `mmseqs2_use_env`
- `mmseqs2_use_filter`
- `mmseqs2_output_dir`



## References

- LossLab documentation: see the LossLab README and loss/refinement docs in the LossLab repository.
- ROCKET YAML parameters: see docs/yaml_parameters.md
