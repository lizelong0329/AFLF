r"""Helper script for setting up the AFRE atom groups and masks."""
# Authors: Zilin Song.

from itertools import groupby

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dssp      import DSSP
from MDAnalysis.analysis.distances import distance_array


dest = "test2_tem1"


def segment_by_secondary_structure(pdb: str) -> tuple[list[np.ndarray], list[str]]:
  r"""Returns the residue indexes of the segments grouped by secondary structures."""
  ss_raw = ''.join(DSSP(mda.Universe(pdb, format='pdb')).run().results.dssp[0])
  print(f"SS string - RAW\n{ss_raw}")
  # 2. relabel E or H segments to loop `-` by visual.
  #     -HHHHHHHHHHHHH---EEEEEEEE-----E-EEE----EEE-HHHHHHHHHHHHHHHHH--------EE---HHH-------HHH-----EEHHHHHHHHHH---HHHHHHHHHHH--HHHHHHHHHH------E------HHH---------EEEHHHHHHHHHHHHH-----HHHHHHHHHHHH--------HHHH-----EEEEEEEE-----EEEEEEEE------EEEEEEEE-----HHHHHHHHHHHHHHHHHH-
  ss = '-HHHHHHHHHHHHH---EEEEEEEE-----EEEEE--------HHHHHHHHHHHHHHHHH---------------------------------HHHHHHHHHH---HHHHHHHHHHH--HHHHHHHHHH----------------------------HHHHHHHHHHHHH-----HHHHHHHHHHHH-----------------EEEEEEEE-----EEEEEEEE------EEEEEEEE-----HHHHHHHHHHHHHHHHHH-'
  segs = list()
  for _ in [''.join(list(_[1])) for _ in groupby(ss)]:
    segs.append(_)
  segs = [''.join(list(_[1])) for _ in groupby(''.join(segs))]
  print(f"SS string - Preprocessed\n{''.join(segs)}")
  # 3. split the preprocessed ss string by segments of identical tokens;.
  rid_sections = np.cumsum(np.asarray([len(_) for _ in segs]))
  rids = np.split(np.arange(len(ss)), indices_or_sections=rid_sections)
  # 4. for each segment, split into groups of residue indexes.
  rid_list, seg_list = [], []
  for s, r in list(zip(segs, rids)):
    assert s[0] in ['E', 'H', '-'], f"Illegal segment flag: `{s[0]}`."
    if s[0] in ['E', ]:               # A. beta-sheets are splitted into segments;
      rid_list.append(r.tolist()); seg_list.append(s[0])
    if s[0] in ['H', ]:               # B. alpha-helix are splitted into segments;
      rid_list.append(r.tolist()); seg_list.append(s[0])
    if s[0] in ['-', ] and len(s)>=7: # C. loops longer than 7 takes the mid-7 residues as one segment.
      mid = len(r.tolist())//2; rid_list.append(r.tolist()[mid-3:mid+4]); seg_list.append(s[0])
  print(f"Segmentation results:")
  for r, s in list(zip(rid_list, seg_list)):
    print(f"{s}: {r}")
  print(f"Number of segments: {len(rid_list)}")
  return rid_list, seg_list

def mask_by_segment_contact(pdb: str, dist_thresh: float) -> list[tuple[int, int, int]]:
  r"""Returns the (N_group, N_group) mask."""
  u = mda.Universe(pdb, format='pdb'); rid = list(range(len(u.residues)))
  atom_groups = [u.select_atoms(f"resid {_+1}:{_+2}") for _ in rid]
  # 1. mask to zero if inter-group-atom distances are all > dist_thresh.
  mask_dist = []
  for i, i_ag in enumerate(atom_groups):
    for j, j_ag in enumerate(atom_groups):
      dist = distance_array(i_ag, j_ag)
      mask_dist.append((i, j, int((dist<=dist_thresh).any())))
  return mask_dist


if __name__ == '__main__':
  pdb = f"./{dest}/aftools_runtime/checkpoint/infer_from_checkpoint/evoformer_result.pdb"
  rid, seg = segment_by_secondary_structure(pdb=pdb)
  m_dist = mask_by_segment_contact(pdb=pdb, dist_thresh=4.5)
  import pickle as pkl
  pkl.dump({'rid': rid, 
            'seg': seg, 
            'mask_dist': m_dist, 
            }, open(f"./{dest}/prep.pkl", 'wb'))