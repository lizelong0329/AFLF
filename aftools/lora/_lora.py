r"""AlphaFoldTools Low-Rank Adaptation (LoRA) facilities."""
# Authors: Zilin Song.


import copy
import typing
import functools

import jax
import jax.numpy as jnp

from aftools import BasePreset
import aftools.utils as _u
import aftools.chkp  as _c
import aftools.geom  as _g


# LoRA mask functions.
def _mask_maximum(x: jnp.ndarray, k: typing.Union[None, int]) -> jnp.ndarray:
  r"""Create a mask such that the largest values in `x` above the `threshold` is masked.
  
    Args:
      x (jnp.ndarray): The tensor to be masked by the generated mask tensor.
      k (Union[None, int]): 
        The top-K maximum elements to be masked out, allowed values are:
          `None`:    The maximum elements are all unmasked.
          `int > 0`: The number of the maximum elements to mask.
    
    Returns:
      mask (jnp.ndarray): The mask tensor to `x`.
  """
  k = 0 if k is None else k
  assert isinstance(k, int) and (0<=k<=x.size), f"Illegal `k={k}`."
  _, inds = jax.lax.top_k(x.flatten(), k=k)
  mask = jnp.ones((x.size, )).at[inds].set(0)
  return jnp.asarray(mask.reshape(*x.shape), dtype=int)


def _mask(reprs:     _u.af.TAFFeatures, 
          mask_to:   typing.Union[list[str], typing.Literal['single', 'msa', 'pair']],
          mask_mink: typing.Union[list[typing.Union[None, int]], None, int], 
          mask_maxk: typing.Union[list[typing.Union[None, int]], None, int],
          lora_masks: _u.af.TAFFeatures = None, 
          ) -> _u.af.TAFFeatures:
  r"""Create the binary mask tensor to mask out the extreme values in `reprs`: 0-masked/1-unmasked.
  
    Args:
      reprs (TAFFeatures): The AF/AF-Multimer checkpoint representations.
      mask_to (Union[list[str], str]): 
        The representation to LoRA-mask: `single`, `msa`, `pair`, or a list of these.
      mask_mink (typing.Union[None, int]): 
        The top-K minimum elements to be masked out, allowed values are:
          `None`:    The minimum elements are all unmasked.
          `int > 0`: The number of the minimum elements to mask.
        If a list is given, the values correspond sequentially for `single|msa` and `pair`.
      mask_maxk (typing.Union[None, int]):
        The top-K maximum elements to be masked out, allowed values are:
          `None`:    The maximum elements are all unmasked.
          `int > 0`: The number of the maximum elements to mask.
        If a list is given, the values correspond sequentially for `single|msa` and `pair`.
      lora_masks (TAFFeatures): The LoRA masks, create a new `dict()` if `None`.
    
    Returns:
      lora_masks (TAFFeatures): The LoRA masks, updated in-place if `not lora_masks is None`.
  """
  if lora_masks is None:
    lora_masks = dict()
  if isinstance(mask_to, list):
    mask_mink = mask_mink if isinstance(mask_mink, list) else [mask_mink for _ in mask_to]
    mask_maxk = mask_maxk if isinstance(mask_maxk, list) else [mask_maxk for _ in mask_to]
    for m_to, m_mink, m_maxk in zip(mask_to, mask_mink, mask_maxk):
      lora_masks = _mask(reprs=reprs, 
                         mask_to  =m_to, 
                         mask_mink=m_mink, 
                         mask_maxk=m_maxk, 
                         lora_masks=lora_masks, )
  elif isinstance(mask_to, str):
    assert mask_to in ['single', 'msa', 'pair'], f"Illegal `mask_to={mask_to}`."
    mask_mink = _mask_maximum(x=-reprs[f'repr_{mask_to}'], k=mask_mink)
    mask_maxk = _mask_maximum(x= reprs[f'repr_{mask_to}'], k=mask_maxk)
    mask      = jnp.asarray((mask_mink+mask_maxk)>1.9, dtype=int)  # residual-stable sum == 2.
    lora_masks.update({f'lora_mask_{mask_to}': mask})
  return lora_masks


def mask_func(infer_from: typing.Literal['structmod', 'evoformer'], 
              apply_lora: typing.Literal['single', 'msa', 'pair', 'both'], 
              mask_mink:  typing.Union[list[typing.Union[None, int]], None, int], 
              mask_maxk:  typing.Union[list[typing.Union[None, int]], None, int],
              ) -> typing.Callable[[_u.af.TAFFeatures, _u.af.TAFFeatures], _u.af.TAFFeatures]:
  r"""Return the Callable function for creating the LoRA masks.
  
    Args:
      infer_from (str): The checkpoint to infer from: `structmod`, `evoformer`.
      apply_lora (str): The representation to apply LoRA: `single`, `msa`, `pair`, `both`.
      mask_mink (Union[list[Union[None, int]], None, int]): 
        The top-K minimum elements to be masked out, allowed values are:
          `None`:    The minimum elements are all unmasked.
          `int > 0`: The number of the minimum elements to mask.
        If a list is given, the values correspond sequentially for `single|msa` and `pair`.
      mask_maxk (Union[list[Union[None, int]], None, int]):
        The top-K maximum elements to be masked out, allowed values are:
          `None`:    The maximum elements are all unmasked.
          `int > 0`: The number of the maximum elements to mask.
        If a list is given, the values correspond sequentially for `single|msa` and `pair`.

    Returns:
      f_mask (Callable):
        The Callable function for creating the LoRA masks.  
        Input:
          reprs      (TAFFeatures): The AF/AF-Multimer checkpoint representations.
          lora_masks (TAFFeatures): The LoRA masks, create a new `dict()` if `None`.
        Output:
          lora_masks (TAFFeatures): The LoRA masks, updated in-place if `not lora_masks is None`.
  """
  mask_to = _c.get_action_to(infer_from=infer_from, apply_lora=apply_lora)
  return functools.partial(_mask, mask_to=mask_to, mask_mink=mask_mink, mask_maxk=mask_maxk)


# LoRA init functions.
def _init(reprs: _u.af.TAFFeatures, 
          init_to:    typing.Union[list[str], typing.Literal['single', 'msa', 'pair']], 
          init_from:  typing.Literal['uniform', 'normal'], 
          init_rank:  int, 
          init_scale: float, 
          key: jax.random.KeyArray, 
          lora_feats: _u.af.TAFFeatures = None, 
          ) -> _u.af.TAFFeatures:
  r"""Initialize the LoRA tensors to the `lora_feats`.
  
    Args:
      reprs (TAFFeatures): The AF/AF-Multimer checkpoint representations.
      init_to (Union[list[str], str]): 
        The representation to LoRA-initialize: `single`, `msa`, `pair`, or a list of these.
      init_from (str): The initializing sampler of LoRA tensors: 'uniform', 'normal'.
      init_rank (int): The tensor rank of the LoRA tensors, must be positive.
      init_scale (float): The scaling factor to the LoRA tensors.
      key (jax.random.KeyArray): The JAX PRNG key.
      lora_feats (TAFFeatures): The LoRA features, create a new `dict()` if `None`.
    
    Returns:
      lora_feats (TAFFeatures): The LoRA features, updated in-place if `not lora_feats is None`.
  """
  if lora_feats is None:
    lora_feats = dict()
  if   isinstance(init_to, list):
    for i_to in init_to:
      lora_feats = _init(reprs=reprs, 
                         init_to   =i_to, 
                         init_from =init_from, 
                         init_rank =init_rank, 
                         init_scale=init_scale, 
                         key=key, 
                         lora_feats=lora_feats, )
  elif isinstance(init_to, str):
    assert init_to   in ['single', 'msa', 'pair'], f"Illegal `init_to={init_to}`."
    assert init_from in ['uniform', 'normal'],     f"Illegal `init_from={init_from}`."
    assert isinstance(init_rank , int  ) and init_rank >0, f"Illegal `init_rank={init_rank}`."
    assert isinstance(init_scale, float) and init_scale>0, f"Illegal `init_scale={init_scale}`."
    shape = reprs[f'repr_{init_to}'].shape
    assert len(shape) in [2, 3], f"Illegal `len(reprs['repr_{init_to}'].shape)={len(shape)}`."
    keys = jax.random.split(key=key, num=len(shape))
    f_sample = {'uniform': _g.rand.uniform, 'normal': _g.rand.normal}.get(init_from, None)
    lora_tensors = {'A': f_sample(key=keys[0], shape=(           shape[0], init_rank))*init_scale,
                    'B': f_sample(key=keys[1], shape=(init_rank, shape[1]           ))*init_scale,
                    } if len(shape)==2 else {
                    'A': f_sample(key=keys[0], shape=(           shape[0], init_rank))*init_scale,
                    'B': f_sample(key=keys[1], shape=(init_rank, shape[1], init_rank))*init_scale,
                    'C': f_sample(key=keys[2], shape=(init_rank, shape[2],          ))*init_scale,
                    } if len(shape)==3 else None
    lora_feats.update({f'lora_feat_{init_to}': lora_tensors})
  return lora_feats


def init_func(infer_from: typing.Literal['structmod', 'evoformer'], 
              apply_lora: typing.Literal['single', 'msa', 'pair', 'both'], 
              init_from:  typing.Literal['uniform', 'normal'], 
              init_rank:  int, 
              init_scale: float, 
              ) -> typing.Callable[[_u.af.TAFFeatures, 
                                    _u.af.TAFFeatures, 
                                    jax.random.KeyArray, ], _u.af.TAFFeatures]:
  r"""Return the Callable function for initializing the LoRA tensors.
  
    Args:
      infer_from (str): The checkpoint to infer from: `structmod`, `evoformer`.
      apply_lora (str): The representation to apply LoRA: `single`, `msa`, `pair`, `both`.
      init_from  (str): The initializing sampler of LoRA tensors: 'uniform', 'normal'.
      init_rank  (int): The tensor rank of the LoRA tensors, must be positive.
      init_scale (float): The scaling factor to the LoRA tensors.

    Returns:
      f_init (Callable):
        The initialize LoRA function.  
        Input:
          reprs (TAFFeatures): The AF/AF-Multimer checkpoint representations.
          key (jax.random.KeyArray): The JAX PRNG key.
          lora_feats (TAFFeatures): The LoRA features, create a new `dict()` if `None`.
        Output:
          lora_feats (TAFFeatures): The LoRA features, updated in-place if `not lora_feats is None`.
  """
  init_to = _c.get_action_to(infer_from=infer_from, apply_lora=apply_lora)
  return functools.partial(_init, 
                           init_to   =init_to, 
                           init_from =init_from, 
                           init_rank =init_rank, 
                           init_scale=init_scale, )


# Representation offsets.
def _offset(reprs:      _u.af.TAFFeatures, 
            lora_feats: _u.af.TAFFeatures, 
            lora_masks: _u.af.TAFFeatures, 
            offset_to:  typing.Union[list[str], typing.Literal['single', 'msa', 'pair']],
            ) -> _u.af.TAFFeatures:
  r"""Offset the `reprs` by the LoRA tensors. This is to ensure that the `reprs` is unchanged after
    executing LoRA at the first time.

    Args:
      reprs      (TAFFeatures): The AF/AF-Multimer checkpoint representations.
      lora_feats (TAFFeatures): The LoRA features.
      lora_masks (TAFFeatures): The LoRA masks.
      offset_to (Union[list[str], str]): 
        The representation to LoRA-offset: `single`, `msa`, `pair`, or a list of these.

    Returns:
      reprs (TAFFeatures): The AF/AF-Multimer checkpoint representations, updated in-place.
  """
  if   isinstance(offset_to, list):
    for o_to in offset_to:
      reprs = _offset(lora_feats=lora_feats, lora_masks=lora_masks, reprs=reprs, offset_to=o_to)
  elif isinstance(offset_to, str):
    assert offset_to in ['single', 'msa', 'pair'], f"Illegal `offset_to={offset_to}`."
    einsum_expr = {'single': 'ai,ib->ab', 'msa': 'ai,ibj,jc->abc', 'pair': 'ai,ibj,jc->abc'}
    V = lora_feats[f'lora_feat_{offset_to}'].values()
    M = lora_masks[f'lora_mask_{offset_to}']
    reprs[f'repr_{offset_to}'] -= (jnp.einsum(einsum_expr.get(offset_to, 'Wrong expr.'), *V) * M)
  return reprs


def offset_func(infer_from: typing.Literal['structmod', 'evoformer'], 
                apply_lora: typing.Literal['single', 'msa', 'pair', 'both'], 
                ) -> typing.Callable[[_u.af.TAFFeatures, 
                                      _u.af.TAFFeatures, 
                                      _u.af.TAFFeatures, ], _u.af.TAFFeatures]:
  r"""Return the Callable function for offsetting the representation by the LoRA tensors and masks.
  
    Args:
      infer_from (str): The checkpoint to infer from: `structmod`, `evoformer`.
      apply_lora (str): The representation to apply LoRA: `single`, `msa`, `pair`, `both`.

    Returns:
      f_offset (Callable):
        The offset representation function.  
        Input:
          lora_feats (TAFFeatures): The LoRA features.
          lora_masks (TAFFeatures): The LoRA masks.
          reprs      (TAFFeatures): The AF/AF-Multimer checkpoint representations.
        Output:
          reprs (TAFFeatures): The AF/AF-Multimer checkpoint representations, updated in-place.
  """
  offset_to = _c.get_action_to(infer_from=infer_from, apply_lora=apply_lora)
  return functools.partial(_offset, offset_to=offset_to)


# wrapper - create the lora_checkpoint.pkl.
def lora_checkpoint(reprs: _u.af.TAFFeatures, 
                    batch: _u.af.TAFFeatures, 
                    infer_from: typing.Literal['structmod', 'evoformer'], 
                    apply_lora: typing.Literal['single', 'msa', 'pair', 'both'], 
                    init_from:  typing.Literal['uniform', 'normal'], 
                    init_rank:  int, 
                    init_scale: float, 
                    mask_mink: typing.Union[None, int], 
                    mask_maxk: typing.Union[None, int], 
                    key: jax.random.KeyArray, 
                    ) -> _u.af.TAFFeatures:
  r"""Create the LoRA checkpoint pickle file.
  
    Args:
      reprs (TAFFeatures): The AF/AF-Multimer checkpoint representations.
      batch (TAFFeatures): The AF/AF-Multimer batch features.
      infer_from (str): The checkpoint to infer from: `structmod`, `evoformer`.
      apply_lora (str): The representation to apply LoRA: `single`, `msa`, `pair`, `both`
      init_from  (str): The initializing sampler of LoRA tensors: 'uniform', 'normal'.
      init_rank  (int): The tensor rank of the LoRA tensors, must be positive.
      init_scale (float): The scaling factor to the LoRA tensors.
      mask_mink (typing.Union[None, int]): 
        The top-K minimum elements to be masked out, allowed values are:
          `None`:    The minimum elements are all unmasked.
          `int > 0`: The number of the minimum elements to mask.
      mask_maxk (typing.Union[None, int]):
        The top-K maximum elements to be masked out, allowed values are:
          `None`:    The maximum elements are all unmasked.
          `int > 0`: The number of the maximum elements to mask.
      key (jax.random.KeyArray): The JAX PRNG key.

    Returns:
      lora_checkpoint_pkl (TAFFeatures): The LoRA checkpoint pickle file, `lora_checkpoint.pkl`.
  """
  # sanity checks.
  assert infer_from in ['structmod', 'evoformer'],        f"Illegal `infer_from={infer_from}`."
  assert apply_lora in ['single', 'msa', 'pair', 'both'], f"Illegal `apply_lora={apply_lora}`."
  # callables.
  f_init  =  init_func(infer_from=infer_from, 
                       apply_lora=apply_lora, 
                       init_from =init_from, 
                       init_rank =init_rank, 
                       init_scale=init_scale, )
  f_mask  =  mask_func(infer_from=infer_from, 
                       apply_lora=apply_lora, 
                       mask_mink=mask_mink, 
                       mask_maxk=mask_maxk, )
  f_offset=offset_func(infer_from=infer_from, apply_lora=apply_lora)
  # lora_masks, lora_feats, and reprs-offset. in-place ops: do not use `reprs`.
  _reprs = copy.deepcopy(reprs)
  _lora_feats = f_init  (reprs=_reprs, key=key)
  _lora_masks = f_mask  (reprs=_reprs)
  _reprs      = f_offset(reprs=_reprs, lora_feats=_lora_feats, lora_masks=_lora_masks)
  # pack to lora_checkpoint.pkl
  lora_checkpoint_pkl ={f'{infer_from}_lora_feats': _lora_feats, 
                        f'{infer_from}_lora_masks': _lora_masks, 
                        f'{infer_from}_reprs': _reprs, 
                                      'batch': copy.deepcopy(batch), }
  return lora_checkpoint_pkl


class LoraCheckpointPreset(BasePreset):
  r"""The LoRA checkpoint preset."""

  def __init__(self):
    r"""Create a LoRA checkpoint function preset."""
    self.init_from: typing.Literal['normal', 'uniform'] = 'uniform'
    r"""The initializing sampler of LoRA tensors: 'uniform', 'normal'. Default: 'uniform'."""
    self.init_rank: int = 8
    r"""The tensor rank of the LoRA tensors, must be positive. Default: 8."""
    self.init_scale: float = 0.1
    r"""The scaling factor to the LoRA tensors. Default: 0.1."""
    self.mask_mink: typing.Union[list[typing.Union[None, int]], None, int] = None
    r"""The top-K minimum elements to be masked out. Default: None.
      Allowed values are:
        `None`:    The minimum elements are all unmasked.
        `int > 0`: The number of the minimum elements to mask.
      If a list is given, the values correspond sequentially for `single|msa` and `pair`.
    """
    self.mask_maxk: typing.Union[list[typing.Union[None, int]], None, int] = None
    r"""The top-K maximum elements to be masked out. Default: None. 
      Allowed values are:
        `None`:    The maximum elements are all unmasked.
        `int > 0`: The number of the maximum elements to mask.
      If a list is given, the values correspond sequentially for `single|msa` and `pair`.
    """
  
  def instantiate(self, 
                  reprs: _u.af.TAFFeatures, 
                  batch: _u.af.TAFFeatures, 
                  infer_from: typing.Literal['structmod', 'evoformer'], 
                  apply_lora: typing.Literal['single', 'msa', 'pair', 'both'], 
                  key: jax.random.KeyArray, 
                  ) -> _u.af.TAFFeatures:
    r"""Create the LoRA checkpoint from this preset.

      Args:
        reprs (TAFFeatures): The AF/AF-Multimer checkpoint representations.
        batch (TAFFeatures): The AF/AF-Multimer batch features.
        infer_from (str): The checkpoint to infer from: `structmod`, `evoformer`.
        apply_lora (str): The representation to apply LoRA: `single`, `msa`, `pair`, `both`.
        key (jax.random.KeyArray): The JAX PRNG key.

      Returns:
        lora_checkpoint_pkl (TAFFeatures): The LoRA checkpoint pickle file, `lora_checkpoint.pkl`.
    """
    return lora_checkpoint(reprs=reprs, 
                           batch=batch, 
                           infer_from=infer_from, 
                           apply_lora=apply_lora, 
                           init_from=self.init_from, 
                           init_rank=self.init_rank, 
                           init_scale=self.init_scale, 
                           mask_mink=self.mask_mink, 
                           mask_maxk=self.mask_maxk, 
                           key=key, )


# LoRA apply functions.
# NOTE (ZS): _lora() is called on inference time and does not check keys or use for-loops.
def _lora(lora_feats: _u.af.TAFFeatures,
          lora_masks: _u.af.TAFFeatures,
          reprs:      _u.af.TAFFeatures,
          lora_to: typing.Literal['single', 'msa', 'pair'],
          ) -> _u.af.TAFFeatures:
  r"""Execute the LoRA adaptation to the `reprs`.
  
    Args:
      lora_feats (TAFFeatures): The LoRA features.
      lora_masks (TAFFeatures): The LoRA masks.
      reprs      (TAFFeatures): The AF/AF-Multimer checkpoint representations.
      lora_to (str): The representation to LoRA-execute: `single`, `msa`, `pair`.
    
    Returns:
      reprs (TAFFeatures): The AF/AF-Multimer checkpoint representations, updated in-place.
  """
  einsum_expr = {'single': 'ai,ib->ab', 'msa': 'ai,ibj,jc->abc', 'pair': 'ai,ibj,jc->abc'}
  feats = lora_feats[f'lora_feat_{lora_to}'].values()
  mask  = lora_masks[f'lora_mask_{lora_to}']
  reprs[f'repr_{lora_to}'] += (mask * jnp.einsum(einsum_expr.get(lora_to, 'Wrong expr.'), *feats))
  return reprs


def _lora_structmod(lora_feats: _u.af.TAFFeatures, 
                    lora_masks: _u.af.TAFFeatures, 
                    reprs:      _u.af.TAFFeatures, 
                    ) -> _u.af.TAFFeatures:
  reprs = _lora(lora_feats=lora_feats, lora_masks=lora_masks, reprs=reprs, lora_to='single')
  reprs = _lora(lora_feats=lora_feats, lora_masks=lora_masks, reprs=reprs, lora_to='pair'  )
  return reprs


def _lora_evoformer(lora_feats: _u.af.TAFFeatures, 
                    lora_masks: _u.af.TAFFeatures, 
                    reprs:      _u.af.TAFFeatures, 
                    ) -> _u.af.TAFFeatures:
  reprs = _lora(lora_feats=lora_feats, lora_masks=lora_masks, reprs=reprs, lora_to='msa' )
  reprs = _lora(lora_feats=lora_feats, lora_masks=lora_masks, reprs=reprs, lora_to='pair')
  return reprs


def apply_func(infer_from: typing.Literal['structmod', 'evoformer'], 
               apply_lora: typing.Literal['single', 'msa', 'pair', 'both'], 
               ) -> typing.Callable[[_u.af.TAFFeatures,
                                     _u.af.TAFFeatures,
                                     _u.af.TAFFeatures, ], _u.af.TAFFeatures]:
  r"""Return the Callable function for applying the LoRA adaptation to `repr`.
  
    Args:
      infer_from (str): The checkpoint to infer from: `structmod`, `evoformer`.
      apply_lora (str): The representation to apply LoRA: `single`, `msa`, `pair`, `both`.

    Returns:
      f_lora (Callable):
        The execute LoRA function.  
        Input:
          lora_feats (TAFFeatures): The LoRA features.
          lora_masks (TAFFeatures): The LoRA masks.
          reprs      (TAFFeatures): The AF/AF-Multimer checkpoint representations.
        Output:
          reprs (TAFFeatures): The AF/AF-Multimer checkpoint representations, updated in-place.
  """
  lora_funcs = {'structmod': {'single': functools.partial(_lora, lora_to='single'), 
                              'pair':   functools.partial(_lora, lora_to='pair'  ), 
                              'both':   _lora_structmod, }, 
                'evoformer': {'msa':  functools.partial(_lora, lora_to='msa' ),
                              'pair': functools.partial(_lora, lora_to='pair'), 
                              'both': _lora_evoformer}, }
  assert infer_from in lora_funcs            .keys(), f"Illegal `infer_from={infer_from}`."
  assert apply_lora in lora_funcs[infer_from].keys(), f"Illegal `apply_lora={apply_lora}`."
  return lora_funcs[infer_from][apply_lora]