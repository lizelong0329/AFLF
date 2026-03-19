r"""AlphaFoldTools Low-Rank Adaptation (LoRA) models for AF."""
# Authors: Zilin Song.


import typing

import aftools.utils as _u
import aftools.chkp  as _c
import aftools.lora  as _l


def lora_infer(lora_feats: _u.af.TAFFeatures, 
               lora_masks: _u.af.TAFFeatures, 
               reprs:      _u.af.TAFFeatures, 
               batch:      _u.af.TAFFeatures, 
               config:     _u.af.TAFConfig, 
               infer_from: typing.Literal['structmod', 'evoformer', 'evoformer_bf16'], 
               apply_lora: typing.Literal['single', 'msa', 'pair', 'both'], 
               ) -> _u.af.TAFResults:
  r"""Infer AF with LoRA.

    Args:
      lora_feats (TAFFeatures): The LoRA features.
      lora_masks (TAFFeatures): The LoRA masks.
      reprs (TAFFeatures): The AF checkpoint representations.
      batch (TAFFeatures): The AF batch features.
      config (TAFConfig):  The AF configurations.
      infer_from (str): The checkpoint to infer from: `structmod`, `evoformer`, `evoformer_bf16`.
      apply_lora (str): The representation to apply LoRA: `single`, `msa`, `pair`, `both`.

    Returns:
      results (TAFResults): The AF output results.
  """
  # NOTE (ZS): 
  #   `reprs` is a global state and cannot be in-place `+=`-ed during JAX runtime, thus the repack,
  #   which would copy all JAX arrays inside the dict since they are immutable.
  reprs = dict(**reprs)
  # callables.
  f_lora  = _l.       apply_func(infer_from=infer_from[:9], apply_lora=apply_lora)
  f_infer = _c.models.infer_func(infer_from=infer_from)
  # inference.
  out_lora  = f_lora (lora_feats=lora_feats, lora_masks=lora_masks, reprs=reprs)
  out_infer = f_infer(reprs=out_lora, batch=batch, config=config)
  return out_infer