r"""AlphaFoldTools Checkpoint modules for AF."""
# Authors: Zilin Song.


import functools

import jax.numpy as jnp
import haiku as hk

import alphafold.model.modules as _af_modules
import alphafold.model.folding as _af_folding

import aftools.utils as _u


MODULE_SCOPE = {'InferForCheckpoint'     : 'alphafold/alphafold_iteration', 
                'InferFromStructmod'     : 'infer_from_structmod', 
                'InferFromEvoformer'     : 'infer_from_evoformer', 
                'InferFromEvoformer_bf16': 'infer_from_evoformer_bf16', }


# infer for the checkpoint.
InferForCheckpoint = _af_modules.AlphaFold
r"""The AF Checkpoint module that infers and tracks checkpoints."""


# infer from the checkpoint - standalone modules.
_StructureModule = functools.partial(_af_folding.StructureModule, compute_loss=False)
r"""The standalone `StructureModule` with `compute_loss=False`."""


class _Evoformer(hk.Module):
  r"""The standalone `Evoformer`."""

  def __init__(self, config: _u.af.TAFConfig, global_config: _u.af.TAFConfig):
    r"""Create a standalone `Evoformer`.
    
      Args:
        config        (TAFConfig): From `config.model.embeddings_and_evoformer`.
        global_config (TAFConfig): From `config.model.global_config`.
    """
    hk.Module.__init__(self, name='evoformer')
    self. c =        config
    self.gc = global_config
  
  def __call__(self, 
               evoformer_input: _u.af.TAFFeatures, 
               evoformer_masks: _u.af.TAFFeatures, 
               is_training: bool, 
               ) -> dict[str, jnp.ndarray]:
    r"""Infers through the evoformer stack and returns the inferred results.

      Args:
        evoformer_input (TAFFeatures): The evoformer input with keys: `msa`, `pair`.
        evoformer_masks (TAFFeatures): The evoformer masks with keys: `msa`, `pair`.
        is_training (bool): If in training mode.
      
      Returns:
        results (dict[str, jnp.ndarray]): The evoformer output with keys: `single`, `pair`.
    """
    # evoformer stack module.
    evoformer_itr = _af_modules.EvoformerIteration(self.c.evoformer, self.gc, is_extra_msa=False)
    def evoformer_fn(x):
      return evoformer_itr(activations=x, masks=evoformer_masks, is_training=is_training)
    evoformer_fn = hk.remat(evoformer_fn) if self.gc.use_remat==True else evoformer_fn
    evoformer_stk = _af_modules.layer_stack.layer_stack(self.c.evoformer_num_block)(evoformer_fn)
    # evoformer stack inference.
    out_evoformer = evoformer_stk(evoformer_input)
    # single activation module & inference.
    out_single = _af_modules.common_modules.Linear(self.c.seq_channel, name='single_activations'
                                                   )(out_evoformer['msa'][0])
    # pack output.
    results = {'single': out_single, 'pair': out_evoformer['pair']}
    return results


class _EvoformerBF16(_Evoformer):
  r"""The standalone `Evoformer` with the jnp.bfloat16 runtime."""

  def __init__(self, config: _u.af.TAFConfig, global_config: _u.af.TAFConfig):
    r"""Create a standalone `Evoformer` with the jnp.bfloat16 runtime.
    
      Args:
        config        (TAFConfig): From `config.model.embeddings_and_evoformer`.
        global_config (TAFConfig): From `config.model.global_config`.
    """
    _Evoformer.__init__(self, config=config, global_config=global_config)

  def __call__(self,
               evoformer_input: _u.af.TAFFeatures, 
               evoformer_masks: _u.af.TAFFeatures, 
               is_training: bool
               ) -> _u.af.TAFFeatures:
    r"""Infers through the evoformer stack and returns the inferred results.

      Args:
        evoformer_input (TAFFeatures): The evoformer input with keys: `msa`, `pair`.
        evoformer_masks (TAFFeatures): The evoformer masks with keys: `msa`, `pair`.
        is_training (bool): If in training mode.
      
      Returns:
        results (dict[str, jnp.ndarray]): The evoformer output with keys: `single`, `pair`.
    """
    with _u.af._af_utils.bfloat16_context():  # params to jnp.bfloat16 if context in jnp.bfloat16.
      # input to jnp.bfloat16.
      for k in evoformer_input.keys(): evoformer_input[k]=evoformer_input[k].astype(jnp.bfloat16)
      for k in evoformer_masks.keys(): evoformer_masks[k]=evoformer_masks[k].astype(jnp.bfloat16)
      # evoformer stack module.
      evoformer_itr = _af_modules.EvoformerIteration(self.c.evoformer, self.gc, is_extra_msa=False)
      def evoformer_fn(x):
        return evoformer_itr(activations=x, masks=evoformer_masks, is_training=is_training)
      evoformer_fn = hk.remat(evoformer_fn) if self.gc.use_remat==True else evoformer_fn
      evoformer_stk = _af_modules.layer_stack.layer_stack(self.c.evoformer_num_block)(evoformer_fn)
      # evoformer stack inference.
      out_evoformer = evoformer_stk(evoformer_input)
      # single activation module & inference.
      out_single = _af_modules.common_modules.Linear(self.c.seq_channel, name='single_activations'
                                                     )(out_evoformer['msa'][0])
      # pack output.
      results = {'single': out_single, 'pair': out_evoformer['pair']}
      # output to jnp.bfloat32.
      for k in results.keys(): results[k] = results[k].astype(jnp.float32)
    return results


# infer from the checkpoint - the 'structmod' checkpoint.
class InferFromStructmod(hk.Module):
  r"""The AF Checkpoint module that infers from the `structmod` checkpoint through:
    - 0. the structure module;
    - 1. the predicted LDDT head;
    - 2. the predicted aligned error head.
  """
  def __init__(self, config: _u.af.TAFConfig):
    r"""Create an AF Checkpoint module that infers from the `structmod` checkpoint.
    
      Args:
        config (TAFConfig): The AF configurations.
    """
    hk.Module.__init__(self, name=MODULE_SCOPE['InferFromStructmod'])
    self. c = config.model
    self.gc = config.model.global_config
  
  def __call__(self, 
               reprs: _u.af.TAFFeatures, 
               batch: _u.af.TAFFeatures,
               is_training: bool, 
               ) -> _u.af.TAFResults:
    r"""Infer with the evoformer representations through the structure module, the pLDDT head, and 
      the pAE head. 
    
      Args:
        reprs (TAFFeatures):
          The representations with keys `repr_single`, `repr_pair`.
        batch (TAFFeatures):
          The batch features with keys `seq_mask`, `aatype`, `residue_index`, `atom14_atom_exists`, 
          `atom37_atom_exists`, `residx_atom37_to_atom14`.
        is_training  (bool): If in training mode.
      
      Returns:
        results (TAFResults): 
          The inference results with keys:
            `structure_module`:        The structure module output.
            `predicted_lddt`:          The pLDDT head output.
            `predicted_aligned_error`: The pAE   head output.
    """
    # module configs.
    config_structmod = self.c.heads.structure_module
    config_plddt     = self.c.heads.predicted_lddt
    config_pae       = self.c.heads.predicted_aligned_error
    # module instances.
    structmod=_StructureModule                     (config=config_structmod, global_config=self.gc)
    plddt    =_af_modules.PredictedLDDTHead        (config=config_plddt    , global_config=self.gc)
    pae      =_af_modules.PredictedAlignedErrorHead(config=config_pae      , global_config=self.gc)
    # unpack 'structmod' checkpoint.
    representations = {'single': reprs['repr_single'], 'pair': reprs['repr_pair']}
    # forward - structure module.
    out_structmod = structmod(representations=representations, batch=batch, is_training=is_training)
    # out_structmod['representations']['structure_module'] is the structmod activation for pLDDT.
    representations['structure_module']=out_structmod.pop('representations').pop('structure_module')
    # forward - confidence heads.
    out_plddt = plddt(representations=representations, batch=batch, is_training=is_training)
    out_pae   = pae  (representations=representations, batch=batch, is_training=is_training)
    # pack to outputs.
    return {'structure_module':        out_structmod, 
            'predicted_lddt':          out_plddt, 
            'predicted_aligned_error': out_pae, }


# infer from the checkpoint - the 'evoformer' checkpoint.
class InferFromEvoformer(hk.Module):
  r"""The AF Checkpoint module that infers from the `evoformer` checkpoint through:
    - 0. the evoformer stack;
    - 1. the structure module;
    - 2. the predicted LDDT head;
    - 3. the predicted aligned error head.
  """

  def __init__(self, config: _u.af.TAFConfig):
    r"""Create an AF Checkpoint inference module from the `evoformer` checkpoint.
    
      Args:
        config (TAFConfig): The AF configurations.
    """
    hk.Module.__init__(self, name=MODULE_SCOPE['InferFromEvoformer'])
    self. c = config.model
    self.gc = config.model.global_config

  def __call__(self, 
               reprs: _u.af.TAFFeatures, 
               batch: _u.af.TAFFeatures,
               is_training: bool, 
               ) -> _u.af.TAFResults:
    r"""Infer with the evoformer input and masks through the evoformer stack, the structure module, 
      the pLDDT head, and the pAE head. 

      Args:
        reprs (TAFFeatures):
          The representations with keys `repr_msa`, `repr_pair`, `mask_msa`, `mask_pair`.
        batch (TAFFeatures):
          The batch features with keys `seq_mask`, `aatype`, `residue_index`, `atom14_atom_exists`, 
          `atom37_atom_exists`, `residx_atom37_to_atom14`.
        is_training (bool): If in training mode.
      
      Returns:
        results (TAFResults): 
          The inference results with keys:
            `structure_module`:        The structure module output.
            `predicted_lddt`:          The pLDDT head output.
            `predicted_aligned_error`: The pAE   head output.
    """
    # module configs.
    config_evoformer = self.c.embeddings_and_evoformer
    config_structmod = self.c.heads.structure_module
    config_plddt     = self.c.heads.predicted_lddt
    config_pae       = self.c.heads.predicted_aligned_error
    # module instances.
    evoformer=_Evoformer                           (config=config_evoformer, global_config=self.gc)
    structmod=_StructureModule                     (config=config_structmod, global_config=self.gc)
    plddt    =_af_modules.PredictedLDDTHead        (config=config_plddt    , global_config=self.gc)
    pae      =_af_modules.PredictedAlignedErrorHead(config=config_pae      , global_config=self.gc)
    # unpack the 'evoformer' checkpoint.
    evoformer_input = {'msa': reprs['repr_msa'], 'pair': reprs['repr_pair']}
    evoformer_masks = {'msa': reprs['mask_msa'], 'pair': reprs['mask_pair']}
    # forward - evoformer.
    representations = evoformer(evoformer_input=evoformer_input, 
                                evoformer_masks=evoformer_masks, 
                                is_training    =is_training, )
    # forward - structure module.
    out_structmod = structmod(representations=representations, batch=batch, is_training=is_training)
    # out_structmod['representations']['structure_module'] is the structmod activation for pLDDT.
    representations['structure_module']=out_structmod.pop('representations').pop('structure_module')
    # forward - confidence heads.
    out_plddt = plddt(representations=representations, batch=batch, is_training=is_training)
    out_pae   = pae  (representations=representations, batch=batch, is_training=is_training)
    # pack to outputs.
    return {'structure_module':        out_structmod, 
            'predicted_lddt':          out_plddt, 
            'predicted_aligned_error': out_pae, }


# infer from the checkpoint - the 'evoformer_bf16' checkpoint.
class InferFromEvoformerBF16(hk.Module):
  r"""The AF Checkpoint module that infers from the `evoformer` checkpoint through:
    - 0. the evoformer stack with the jnp.bfloat16 runtime;
    - 1. the structure module;
    - 2. the predicted LDDT head;
    - 3. the predicted aligned error head.
  """

  def __init__(self, config: _u.af.TAFConfig):
    r"""Create an AF Checkpoint inference module from the `evoformer` checkpoint.
    
      Args:
        config (TAFConfig): The AF configurations.
    """
    hk.Module.__init__(self, name=MODULE_SCOPE['InferFromEvoformer_bf16'])
    self. c = config.model
    self.gc = config.model.global_config
    
  def __call__(self, 
               reprs: _u.af.TAFFeatures, 
               batch: _u.af.TAFFeatures,
               is_training: bool, 
               ) -> _u.af.TAFResults:
    r"""Infer with the evoformer input and masks through the evoformer stack with the jnp.bfloat16
      runtime, the structure module, the pLDDT head, and the pAE head. 

      Args:
        reprs (TAFFeatures):
          The representations with keys `repr_msa`, `repr_pair`, `mask_msa`, `mask_pair`.
        batch (TAFFeatures):
          The batch features with keys `seq_mask`, `aatype`, `residue_index`, `atom14_atom_exists`, 
          `atom37_atom_exists`, `residx_atom37_to_atom14`.
        is_training (bool): If in training mode.
      
      Returns:
        results (TAFResults): 
          The inference results with keys:
            `structure_module`:        The structure module output.
            `predicted_lddt`:          The pLDDT head output.
            `predicted_aligned_error`: The pAE   head output.
    """
    # module configs.
    config_evoformer = self.c.embeddings_and_evoformer
    config_structmod = self.c.heads.structure_module
    config_plddt     = self.c.heads.predicted_lddt
    config_pae       = self.c.heads.predicted_aligned_error
    # module instances.
    evoformer=_EvoformerBF16                       (config=config_evoformer, global_config=self.gc)
    structmod=_StructureModule                     (config=config_structmod, global_config=self.gc)
    plddt    =_af_modules.PredictedLDDTHead        (config=config_plddt    , global_config=self.gc)
    pae      =_af_modules.PredictedAlignedErrorHead(config=config_pae      , global_config=self.gc)
    # unpack the 'evoformer' checkpoint.
    evoformer_input = {'msa': reprs['repr_msa'], 'pair': reprs['repr_pair']}
    evoformer_masks = {'msa': reprs['mask_msa'], 'pair': reprs['mask_pair']}
    # forward - evoformer.
    representations = evoformer(evoformer_input=evoformer_input, 
                                evoformer_masks=evoformer_masks, 
                                is_training    =is_training, )
    # forward - structure module.
    out_structmod = structmod(representations=representations, batch=batch, is_training=is_training)
    # out_structmod['representations']['structure_module'] is the structmod activation for pLDDT.
    representations['structure_module']=out_structmod.pop('representations').pop('structure_module')
    # forward - confidence heads.
    out_plddt = plddt(representations=representations, batch=batch, is_training=is_training)
    out_pae   = pae  (representations=representations, batch=batch, is_training=is_training)
    # pack to outputs.
    return {'structure_module':        out_structmod, 
            'predicted_lddt':          out_plddt, 
            'predicted_aligned_error': out_pae, }