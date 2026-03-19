r"""AlphaFoldTools Checkpoint models for AF."""
# Authors: Zilin Song.


import typing
import functools

import jax
import haiku as hk

from aftools import BaseModel
import aftools.utils as _u
import aftools.chkp  as _c


def _infer_for_checkpoint(batch: _u.af.TAFFeatures, config: _u.af.TAFConfig) -> _u.af.TAFResults:
  mod = _c.modules.InferForCheckpoint(config=config.model)
  results = mod(batch=batch, 
                is_training=False, 
                compute_loss=False, 
                ensemble_representations=False, 
                return_representations  =False, )
  return results


class InferForCheckpointModel(BaseModel):
  r"""The AF Checkpoint model that infers and tracks the checkpoints."""

  def __init__(self, config: _u.af.TAFConfig, params: _u.af.TAFParams, key: jax.random.KeyArray):
    r"""Create an AF Checkpoint model that infers and tracks the checkpoints.
    
      Args:
        config (TAFConfig): The AF configurations.
        params (TAFParams): The AF parameters.
        key (jax.random.KeyArray): The JAX PRNG Key.
    """
    BaseModel.__init__(self, config=config, params=params, key=key, multimer_mode=False)

  def compile(self, 
              evoformer_checkpoint: bool, 
              structmod_checkpoint: bool, 
              jit_predict: bool, 
              ) -> None:
    r"""Compile the model for inference.
    
      Args:
        evoformer_checkpoint (bool): If record the `evoformer` checkpoint.
        structmod_checkpoint (bool): If record the `structmod` checkpoint.
        jit_predict (bool): If JIT-compile the `predict` function.
    """
    # config.
    self.config.update_from_flattened_dict({'model.global_config.aftools': {}, })
    self.config.update_from_flattened_dict({
      'model.global_config.aftools.chkp_is_last_recycling':    False, 
      'model.global_config.aftools.chkp_evoformer_checkpoint': evoformer_checkpoint, 
      'model.global_config.aftools.chkp_structmod_checkpoint': structmod_checkpoint, })
    # params.
    module_scope = _c.modules.MODULE_SCOPE['InferForCheckpoint']
    self.params = _u.af.prefix_params(params=self.params, prefix=module_scope)
    # pred function.
    f_pred = functools.partial(_infer_for_checkpoint, config=self.config)
    self.f_pred = (jax.jit(hk.transform(f=f_pred).apply if jit_predict else 
                           hk.transform(f=f_pred).apply)                   )

  def predict(self, features_pkl: _u.af.TAFFeatures) -> _u.af.TAFResults:
    r"""Predict with the model.
    
      Args:
        features_pkl (TAFFeatures): From `alphafold.model.features.np_example_to_features()`.
    
      Returns:
        results (TAFResults): The AF outputs.
    """
    results = self.f_pred(self.params, self.key, features_pkl)
    return results


def _infer_from_structmod(reprs:  _u.af.TAFFeatures, 
                          batch:  _u.af.TAFFeatures, 
                          config: _u.af.TAFConfig, 
                          ) -> _u.af.TAFResults:
  mod            = _c.modules.InferFromStructmod   (config=config)
  mod_confidence = _c.AFConfidenceModule(config=config)
  results            = mod(reprs=reprs, batch=batch, is_training=False)
  results_confidence = mod_confidence(results=results, batch=batch)
  results.update({'confidence': results_confidence})
  return results


def _infer_from_evoformer(reprs:  _u.af.TAFFeatures, 
                          batch:  _u.af.TAFFeatures, 
                          config: _u.af.TAFConfig, 
                          ) -> _u.af.TAFResults:
  mod            = _c.modules.InferFromEvoformer   (config=config)
  mod_confidence = _c.AFConfidenceModule(config=config)
  results            = mod(reprs=reprs, batch=batch, is_training=False)
  results_confidence = mod_confidence(results=results, batch=batch)
  results.update({'confidence': results_confidence})
  return results


def _infer_from_evoformer_bf16(reprs:  _u.af.TAFFeatures, 
                               batch:  _u.af.TAFFeatures, 
                               config: _u.af.TAFConfig, 
                               ) -> _u.af.TAFResults:
  mod            = _c.modules.InferFromEvoformerBF16(config=config)
  mod_confidence = _c.AFConfidenceModule(config=config)
  results            = mod(reprs=reprs, batch=batch, is_training=False)
  results_confidence = mod_confidence(results=results, batch=batch)
  results.update({'confidence': results_confidence})
  return results


def infer_func(infer_from: typing.Literal['structmod', 'evoformer', 'evoformer_bf16'], 
               ) -> typing.Callable[[_u.af.TAFFeatures, 
                                     _u.af.TAFFeatures, 
                                     _u.af.TAFConfig, ], _u.af.TAFResults]:
  r"""Get the checkpoint inference functions.
  
    Args:
      infer_from (str): The checkpoint to infer from: `structmod`, `evoformer`, `evoformer_bf16`.
    
    Returns:
      f_infer (Callable):
        The checkpoint inference function.  
        Input:
          reprs  (TAFFeatures): The AF checkpoint representations.
          batch  (TAFFeatures): The AF batch features.
          config (TAFConfig):   The AF configurations.
        Output:
          results (TAFResults): The AF outputs.
  """
  infer_funcs = {'structmod':      _infer_from_structmod, 
                 'evoformer':      _infer_from_evoformer, 
                 'evoformer_bf16': _infer_from_evoformer_bf16, }
  assert infer_from in infer_funcs.keys(), f"Illegal `infer_from={infer_from}`."
  return infer_funcs[infer_from]


class InferFromCheckpointModel(BaseModel):
  r"""The AF Checkpoint model that infers from the checkpoints."""

  def __init__(self, config: _u.af.TAFConfig, params: _u.af.TAFParams, key: jax.random.KeyArray):
    r"""Create an AF Checkpoint model that infers from the checkpoints.
    
      Args:
        config (TAFConfig): The AF configurations.
        params (TAFParams): The AF parameters.
        key (jax.random.KeyArray): The JAX PRNG Key.
    """
    BaseModel.__init__(self, config=config, params=params, key=key, multimer_mode=False)

  def compile(self, 
              batch: _u.af.TAFFeatures, 
              infer_from: typing.Literal['structmod', 'evoformer', 'evoformer_bf16'], 
              jit_predict: bool
              ) -> None:
    r"""Compile the model for inference.
    
      Args:
        batch (TAFFeatures): The AF batch features.
        infer_from (str): The checkpoint to infer from: `structmod`, `evoformer`, `evoformer_bf16`.
        jit_predict (bool): If JIT-compile the `predict` function.
    """
    # params.
    module_scope = _c.modules.MODULE_SCOPE[f'InferFrom{infer_from.capitalize()}']
    self.params = _u.af.prefix_params(params=self.params, prefix=module_scope)
    # pred function.
    f_pred = functools.partial(infer_func(infer_from=infer_from), 
                                  batch =batch, 
                                  config=self.config, )
    self.f_pred = (jax.jit(hk.transform(f=f_pred).apply) if jit_predict else 
                           hk.transform(f=f_pred).apply                     )
  
  def predict(self, reprs: _u.af.TAFFeatures) -> _u.af.TAFResults:
    r"""Predict with the model.
    
      Args:
        reprs (TAFFeatures): The AF checkpoint representations.
    
      Returns:
        results (TAFResults): The AF outputs.
    """
    results = self.f_pred(self.params, self.key, reprs)
    return results