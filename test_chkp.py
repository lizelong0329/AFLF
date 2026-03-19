r"""AFRE checkpoint inference."""
# Authors: Zilin Song.


if not __name__ == '__main__':
  raise RuntimeError(f"{__file__} is not importable.")


import aftools.utils        as _u
import aftools.chkp.runners as _r


dest = 'test2_tem1'


# Create the runner for logging the checkpoints.
config = _u.af.AFConfig(model_name='model_1_ptm')
params = _u.af.AFParams(model_name='model_1_ptm')
runner = _r.InferForCheckpointRunner(project_dir=f'./{dest}')
runner.execute(config=config, 
               params=params, 
               evoformer_checkpoint=True, 
               structmod_checkpoint=True, 
               features_pkl=_u.io.pkl.load(file=f'./{dest}/features.pkl'))

# Load the checkpoint and infer from the checkpoint..
checkpoint_pkl = _u.io.pkl.load(file=f'./{dest}/aftools_runtime/checkpoint/infer_for_checkpoint/checkpoint.pkl')
runner = _r.InferFromCheckpointRunner(project_dir=f'./{dest}')
runner.execute(config=config, 
               params=params, 
               infer_from='evoformer_bf16',   # in jax.numpy.bfloat16
               checkpoint_pkl=checkpoint_pkl, )