r"""AFRE test for tem1."""
# Authors: Zilin Song.


if not __name__ == '__main__':
  raise RuntimeError(f"{__file__} is not importable.")


import optax
import jax.numpy as jnp
jnp.set_printoptions(precision=6, threshold=jnp.inf)

import aftools.utils as _u
import aftools.afre  as _re

dest = 'test2_tem1'

config = _u.af.AFConfig(model_name='model_1_ptm')
params = _u.af.AFParams(model_name='model_1_ptm')
checkpoint_pkl = _u.io.pkl.load(f'./{dest}/aftools_runtime/checkpoint/infer_for_checkpoint/checkpoint.pkl')

# Region selections. ===============================================================================
prep = _u.io.pkl.load(file=f'./{dest}/prep.pkl')
nres = checkpoint_pkl['batch']['aatype'].shape[0] # num. residues.

# rigidbody. ---------------------------------------------------------------------------------------
rigidbody_rids = [[], ]  # first one is for all 'E'.
for (r, s) in zip(prep['rid'], prep['seg']):
  if s=='H': rigidbody_rids   .append(r)
  if s=='E': rigidbody_rids[0].extend(r)
rigidbody_indexers0 = [_u.selection.AtomIndexerByAtomType(_, ['N', 'CA', 'C', 'O', 'CB']) for _ in rigidbody_rids]
rigidbody_indexers1 = [_u.selection.AtomIndexerByCystineBond(cys0=51, cys1=97), ]
rigidbody_indexers2 = [_u.selection.AtomIndexerByProlineRing(pro=_) for _ in [1, 36, 41, 81, 119, 141, 148, 157, 193, 200, 225, 229]]
rigidbody_indexers3 = [_u.selection.AtomIndexerByPeptideBond(residue_index=_) for _ in range(nres-1)]
rigidbody_indexers  =  _u.selection.AtomIndexerGroup().extend(rigidbody_indexers0
                                                     ).extend(rigidbody_indexers1
                                                     ).extend(rigidbody_indexers2
                                                     ).extend(rigidbody_indexers3)

# cistogram. ---------------------------------------------------------------------------------------
cistogram_rids  = jnp.arange(nres, dtype=int).reshape(-1, 1).tolist()
cistogram_indexers = [_u.selection.AtomIndexerByAtomType(_, _u.af.AFAtomTypes) for _ in cistogram_rids]
cistogram_indexers =  _u.selection.AtomIndexerGroup().extend(cistogram_indexers)
cistogram_mask     = 1 - jnp.sum(jnp.asarray([jnp.eye(nres, k=_) for _ in range(-4, 5)]), axis=0)

# anchoring and repelling. -------------------------------------------------------------------------
common_rids = [[], ]  # first one is for all 'E'.
for (r, s) in zip(prep['rid'], prep['seg']):
  if s=='H' or s=='-': common_rids   .append(r)
  if s=='E':           common_rids[0].extend(r)
common_mask = jnp.zeros((len(common_rids), len(common_rids)))
for (i, j, m) in prep['mask_dist']:
  # to zero if i, j locate neighboring segments.
  for igroup, irids in enumerate(prep['rid']):
    for jgroup, jrids in enumerate(prep['rid']):
      if i in irids and j in jrids and abs(igroup-jgroup)<=1:
        m = 0
  if m!=1: continue
  # take dist mask.
  for igroup, irids in enumerate(common_rids):
    for jgroup, jrids in enumerate(common_rids):
      if i in irids and j in jrids and igroup!=jgroup:
        common_mask = common_mask.at[igroup, jgroup].set(m)

anchoring_indexers = [_u.selection.AtomIndexerByAtomType(_, ['N', 'CA', 'C', 'O', 'CB']) for _ in common_rids]
anchoring_indexers =  _u.selection.AtomIndexerGroup().extend(anchoring_indexers)
anchoring_mask     = jnp.copy(common_mask)

repelling_indexers = [_u.selection.AtomIndexerByAtomType(_, ['N', 'CA', 'C', 'O', 'CB']) for _ in common_rids]
repelling_indexers =  _u.selection.AtomIndexerGroup().extend(repelling_indexers)
repelling_mask     = jnp.copy(common_mask)


# presets. =========================================================================================
preset = _re.runners.AFRERunnerPreset()

# loss weights.
preset.af_confidence_loss_spec.weight_plddt      = 0.1
preset.af_confidence_loss_spec.weight_ptm        = 0.

preset.afre_geometry_loss_spec.rigidbody.weight  = 0.5
preset.afre_geometry_loss_spec.cistogram.weight  = 0.5
preset.afre_geometry_loss_spec.anchoring.weight  = 3.0
preset.afre_geometry_loss_spec.repelling.weights = 0.1 * jnp.ones(40)

# loss dist_mask.
preset.afre_geometry_loss_spec.cistogram.mask = cistogram_mask
preset.afre_geometry_loss_spec.anchoring.mask = anchoring_mask
preset.afre_geometry_loss_spec.repelling.mask = repelling_mask

# loss dropouts.
preset.afre_geometry_loss_spec.rigidbody.dropout_p = 0.2
preset.afre_geometry_loss_spec.cistogram.dropout_p = 0.2
preset.afre_geometry_loss_spec.anchoring.dropout_p = 0.0
preset.afre_geometry_loss_spec.repelling.dropout_p = 0.5

# anchoring specs.
preset.afre_geometry_loss_spec.anchoring.thresh_lower = - 1.*jnp.ones_like(anchoring_mask)
preset.afre_geometry_loss_spec.anchoring.thresh_upper =  10.*jnp.ones_like(anchoring_mask)

# cistogram breaks.
# preset.afre_geometry_loss_spec.cistogram.gauss_breaks = jnp.linspace(1, 64, 64) # 1: sudden jumps.
preset.afre_geometry_loss_spec.cistogram.gauss_breaks = jnp.linspace(2, 64, 32)
preset.afre_geometry_loss_spec.cistogram.gauss_rwidth = 4.

# repelling specs.
preset.afre_geometry_loss_spec.repelling.gauss_rwidth = 4.

# runtime.
preset.num_steps = 10000    # total number of steps, trajectories are save per step.
preset.save_freq = 1000     # saves the restart file.
preset.repelling_pushstack_freq = 10          # pushes a new repellent state.
preset.repelling_pushstack_pert = 0.1         # the distant perturbation to pushed repellent states.
preset.repelling_diversify_freq = 20          # updates the diversifying weights.
preset.repelling_diversify_rtau = 10.         # diversity reciprocal tempering.
preset.repelling_diversify_apply_to = 'cent'  # diversify per centroid.

# optimizer.
preset.optax_optimizer = optax.adam(learning_rate=.003)

# create and execute the AFLF runner.
runner = preset.instantiate(project_dir=f'./{dest}')
runner.execute(config=config, 
               params=params, 
               infer_from='evoformer_bf16', 
               apply_lora='msa', 
               checkpoint_pkl=checkpoint_pkl, 
               anchoring_indexers=anchoring_indexers, 
               cistogram_indexers=cistogram_indexers, 
               repelling_indexers=repelling_indexers, 
               rigidbody_indexers=rigidbody_indexers, )