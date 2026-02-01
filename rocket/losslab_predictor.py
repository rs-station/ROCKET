"""Prediction utilities for LossLab integration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rocket import coordinates as rk_coordinates


@dataclass
class PredictorConfig:
    """Configuration for OpenFold predictor wrapper."""

    feature_key: str = "msa_feat"
    init_recycles: int = 1
    iter_recycles: int = 1


class OpenFoldPredictor:
    """Prediction callback that returns [N, 4]: xyz + confidence."""

    def __init__(
        self,
        model,
        features,
        features_backup,
        pdb_obj,
        config: PredictorConfig | None = None,
        bias: bool = True,
    ) -> None:
        self.model = model
        self.features = features
        self.features_backup = features_backup
        self.pdb_obj = pdb_obj
        self.config = config or PredictorConfig()
        self.prevs = None
        self._logged_bias_stats = False
        self.bias = bias

    def __call__(self) -> torch.Tensor:
        """Return [N, 4] tensor: [x, y, z, confidence]."""
        self.features[self.config.feature_key] = self.features_backup.detach().clone()

        init_recycles = self.config.init_recycles
        iter_recycles = self.config.iter_recycles
        if self.prevs is None:
            outputs, self.prevs = self.model(
                self.features,
                [None, None, None],
                num_iters=init_recycles,
                bias=False,
            )
            self.features[self.config.feature_key] = (
                self.features_backup.detach().clone()
            )
            self.prevs = [p.detach() for p in self.prevs]
            deep_copied_prevs = [p.clone().detach() for p in self.prevs]
            outputs, _ = self.model(
                self.features,
                deep_copied_prevs,
                num_iters=iter_recycles,
                bias=self.bias,
            )
        else:
            deep_copied_prevs = [p.clone().detach() for p in self.prevs]
            outputs, _ = self.model(
                self.features,
                deep_copied_prevs,
                num_iters=iter_recycles,
                bias=self.bias,
            )

        xyz_orth_sfc, plddts = rk_coordinates.extract_allatoms(
            outputs,
            self.features,
            self.pdb_obj.cra_name,
        )
        return torch.cat([xyz_orth_sfc, plddts.unsqueeze(-1)], dim=-1)
