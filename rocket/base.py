"""
Include modified subclasses of AlphaFold
"""

import os
import re

import torch
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.script_utils import get_model_basename

# from openfold.utils.tensor_utils import tensor_tree_map
from rocket.utils import get_params_path, tensor_tree_map


class MSABiasAFv1(AlphaFold):
    """
    AlphaFold with trainable bias in MSA space
    """

    def __init__(
        self,
        config,
        preset,
        params_root=None,
        use_deepspeed_evo_attention=True,
    ):
        super().__init__(config)

        if params_root is None:
            params_root = get_params_path()

        # AlphaFold params
        params_path = os.path.join(params_root, f"params_{preset}.npz")
        model_basename = get_model_basename(params_path)
        model_version = "_".join(model_basename.split("_")[1:])
        import_jax_weights_(self, params_path, version=model_version)
        config.globals.use_deepspeed_evo_attention = use_deepspeed_evo_attention
        # print("DEEPSPEED IS ", config.globals.use_deepspeed_evo_attention)
        self.eval()  # without this, dropout enabled

        # self.train()
        # for m in self.modules():
        #    if m.__class__.__name__.startswith("Dropout"):
        #        m.eval()

    def freeze(self, skip_str=None):
        """
        freeze AF2 parameters, skip those parameters with str match
        """
        if skip_str is None:
            for params in self.parameters():
                params.requires_grad_(False)
        else:
            for name, params in self.named_parameters():
                if re.match(skip_str, name) is None:
                    params.requires_grad_(False)

    def unfreeze(self, skip_str=None):
        """
        unfreeze AF2 parameters, skip those parameters with str match
        """
        if skip_str is None:
            for params in self.parameters():
                params.requires_grad_(True)
        else:
            for name, params in self.named_parameters():
                if re.match(skip_str, name) is None:
                    params.requires_grad_(True)

    def _bias(self, feats):
        feats["msa_feat"][:, :, 25:48] = (
            feats["msa_feat"][:, :, 25:48] + feats["msa_feat_bias"]
        )
        return feats

    def iteration(self, feats, prevs, _recycle=True, bias=True):
        if bias:
            feats = self._bias(feats)
        return super().iteration(feats, prevs, _recycle)

    def forward(self, batch, prevs=None, num_iters=1, bias=True):
        if prevs is None:
            prevs = [None, None, None]
        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]  # noqa: E731, B023
            feats = tensor_tree_map(fetch_cur_batch, batch)

            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter and torch.is_autocast_enabled():
                    # Sidestep AMP bug (PyTorch issue #65766)
                    torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev, _ = self.iteration(
                    feats, prevs, _recycle=(num_iters > 1), bias=bias
                )

                if not is_final_iter:
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev

        # Run auxiliary heads
        outputs.update(self.aux_heads(outputs))

        return outputs, [m_1_prev, z_prev, x_prev]


class MSABiasAFv3(MSABiasAFv1):
    """
    AlphaFold with trainable bias + trainable linear combination in MSA space
    """

    def _bias(self, feats):
        feats["msa_feat"][:, :, 25:48] = (
            feats["msa_feat"][:, :, 25:48].clone() * feats["msa_feat_weights"]
            + feats["msa_feat_bias"]
        )
        return feats


class TemplateBiasAF(MSABiasAFv1):
    """
    AlphaFold with trainable bias in template representation
    """

    def _bias(self, feats):
        # TODO: make sure the following operations are valid,
        # Values in feature have to be mapped into -1.0 - 1.0

        feats["template_torsion_angles_sin_cos"] = (
            feats["template_torsion_angles_sin_cos"].clone()
            + feats["template_torsion_angles_sin_cos_bias"]
        )
        return feats
