r"""AlphaFoldTools Checkpoint runners for AF."""
# Authors: Zilin Song.


import os
import typing

import numpy as np

import alphafold.model.features as _af_features

from aftools import BaseRunner, Runtime
import aftools.utils as _u
import aftools.chkp  as _c


class InferForCheckpointRunner(BaseRunner):
  r"""The AF Checkpoint runner that infers and tracks the checkpoints."""

  def __init__(self, project_dir: str = os.getcwd()):
    r"""Create an AF Checkpoint runner that infers and tracks the checkpoints.
    
      Args:
        project_dir (str): The project directory for the runner I/O.
    """
    BaseRunner.__init__(self, project_dir=project_dir, runner_name=r'chkp.infer_for_checkpoint')
  
  def execute(self, 
              config: _u.af.TAFConfig, 
              params: _u.af.TAFParams, 
              evoformer_checkpoint: bool, 
              structmod_checkpoint: bool, 
              features_pkl: _u.af.TAFFeatures, 
              features_seed: typing.Union[None, int] = None, 
              ) -> None:
    r"""Execute the runner.
    
      Args:
        config (TAFConfig): The AF configurations.
        params (TAFParams): The AF parameters.
        evoformer_checkpoint (bool): If record the `evoformer` checkpoint.
        structmod_checkpoint (bool): If record the `structmod` checkpoint.
        features_pkl (TAFFeatures): From `np.load('features.pkl', allow_pickle=True)`.
        features_seed (Union[None, int]): The random seed for featurizing `features_pkl`.
    """
    # The random seed for featurization.
    if features_seed is None:
      features_seed = np.random.randint(low=-99999, high=100000)
    assert isinstance(features_seed, int), f"Illegal `features_seed={features_seed}`."
    with Runtime(whoami=self.runner_name, working_dir=self.working_dir) as RT:
      # kernels.
      mod = _c.models.InferForCheckpointModel(config=config, params=params, key=self.key)
      mod.compile(evoformer_checkpoint=evoformer_checkpoint, 
                  structmod_checkpoint=structmod_checkpoint, 
                  jit_predict=False, )  # Do not jit.
      features_pkl = _af_features.np_example_to_features(np_example =features_pkl, 
                                                         config     =mod.config, 
                                                         random_seed=features_seed, )
      results = mod.predict(features_pkl=features_pkl)
      results = _u.af.cast_jnp_to_np(dict_jnp=results)
      # outputs.
      out_batch = {'seq_mask':                results.pop('aftools_batch_seq_mask'               ), 
                   'aatype':                  results.pop('aftools_batch_aatype'                 ), 
                   'residue_index':           results.pop('aftools_batch_residue_index'          ), 
                   'atom14_atom_exists':      results.pop('aftools_batch_atom14_atom_exists'     ), 
                   'atom37_atom_exists':      results.pop('aftools_batch_atom37_atom_exists'     ),
                   'residx_atom37_to_atom14': results.pop('aftools_batch_residx_atom37_to_atom14'),}
      # outputs - checkpoint requested.
      out_checkpoint = {}
      if evoformer_checkpoint==True:
        out_checkpoint.update({'evoformer_reprs': {
                                  'repr_msa':  results.pop('aftools_evoformer_repr_msa' ), 
                                  'repr_pair': results.pop('aftools_evoformer_repr_pair'), 
                                  'mask_msa':  results.pop('aftools_evoformer_mask_msa' ), 
                                  'mask_pair': results.pop('aftools_evoformer_mask_pair'), }, })
      if structmod_checkpoint==True:
        out_checkpoint.update({'structmod_reprs': {
                                  'repr_single': results.pop('aftools_structmod_repr_single'), 
                                  'repr_pair':   results.pop('aftools_structmod_repr_pair'  ), }, })
      # results.
      RT.logger.info("Output results ...")
      RT.io.pdb.dump_af(file=RT.todir('result.pdb'), results=results, batch=out_batch)
      RT.io.pkl.dump   (file=RT.todir('result.pkl'), to_dump=results)
      # outputs - checkpoint not requested.
      if not (evoformer_checkpoint or structmod_checkpoint):
        ## no aftools features or checkpoints in results.
        assert not 'aftools_evoformer_repr_msa'    in results.keys()
        assert not 'aftools_evoformer_repr_pair'   in results.keys()
        assert not 'aftools_evoformer_mask_msa'    in results.keys()
        assert not 'aftools_evoformer_mask_pair'   in results.keys()
        assert not 'aftools_structmod_repr_single' in results.keys()
        assert not 'aftools_structmod_repr_pair'   in results.keys()
        RT.logger.info("No checkpoint required ...")
        return
      # outputs - checkpoint requested.
      out_checkpoint.update({'batch': out_batch})
      RT.logger.info("Output checkpoint ...")
      RT.io.pkl.dump(file=RT.todir('checkpoint.pkl'), to_dump=out_checkpoint)


class InferFromCheckpointRunner(BaseRunner):
  r"""The AF Checkpoint runner that infers from the checkpoints."""

  def __init__(self, project_dir: str = os.getcwd()):
    r"""Create an AF Checkpoint runner that infers from the checkpoints.
    
      Args:
        project_dir (str): The project base directory for the runner I/O.
    """
    BaseRunner.__init__(self, project_dir=project_dir, runner_name=r'chkp.infer_from_checkpoint')
  
  def execute(self,
              config: _u.af.TAFConfig, 
              params: _u.af.TAFParams, 
              infer_from: typing.Literal['structmod', 'evoformer', 'evoformer_bf16'],
              checkpoint_pkl: _u.af.TAFFeatures, 
              ) -> None:
    r"""Execute the runner.
    
      Args:
        config (TAFConfig): The AF configurations.
        params (TAFParams): The AF parameters.
        infer_from (str): The checkpoint to infer from: `structmod`, `evoformer`, `evoformer_bf16`.
        checkpoint_pkl (TAFFeatures): The checkpoint pickle file, `checkpoint.pkl`.
    """
    with Runtime(whoami=self.runner_name, working_dir=self.working_dir) as RT:
      batch, reprs = checkpoint_pkl['batch'], checkpoint_pkl[f'{infer_from[:9]}_reprs']
      # kernels.
      mod = _c.models.InferFromCheckpointModel(config=config, params=params, key=self.key)
      mod.compile(batch=batch, infer_from=infer_from, jit_predict=False)  # Do not jit.
      results = mod.predict(reprs=reprs)
      results = _u.af.cast_jnp_to_np(dict_jnp=results)
      # outputs.
      RT.logger.info("Output results ...")
      RT.io.pdb.dump_af(file=RT.todir(f'{infer_from}_result.pdb'), results=results, batch=batch)
      RT.io.pkl.dump   (file=RT.todir(f'{infer_from}_result.pkl'), to_dump=results)