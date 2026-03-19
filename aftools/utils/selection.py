r"""AlphaFoldTools utilities for atom selection."""
# Authors: Zilin Song.


import copy
import typing

import numpy as np

import aftools.utils as _u


class AtomIndexer:
  r"""Atom index look-ups."""

  def __init__(self, lookups: dict[int, np.ndarray]):
    r"""Create an Atom index look-ups.
    
      Args:
        lookups (dict[int, np.ndarray): The look-up table as `{residue_index: [AF_atom_indexes]}`.
    """
    self.lookups = copy.deepcopy(lookups)

  def mm_indexes(self, top: _u.mm.TMMTop) -> np.ndarray:
    r"""Returns the atom indexes selected from OpenMM topologies.
    
      Args:
        top (TMMTop): The OpenMM topology.

      Returns:
        indexes (np.ndarray): The selected atom indexes (N, ) from the full indexing (N_atoms, ).
    """
    # The ugly for-nesting is required to match the ordering of atoms selected from AF.
    mm_indexes = list()
    for   rid in list(self.lookups.keys()):
      for aid in list(self.lookups.get(rid, None)):
        assert not aid is None, f"Illegal `lookups[{rid}]=None` in atom look-ups."
        for mm_atom, mm_aid in list(zip( list(top.atoms()), list(np.arange(top.getNumAtoms())) )):
          if mm_atom.residue.index==rid and mm_atom.name==_u.af.AFIndexAtomType[aid]:
            mm_indexes.append(mm_aid)
    return np.asarray(mm_indexes)

  def af_indexes(self, batch: _u.af.TAFFeatures, is_multimer: bool) -> np.ndarray:
    r"""Returns the atom indexes selected from AF/AF-Multimer batch features.
    
      Args:
        batch       (TMMTop): The AF/AFMultimer batch features.
        is_multimer (bool):   If is AFMultimer.

      Returns:
        indexes (np.ndarray): The selected atom indexes (N, ) from the full indexing (N_res*37, ).
    """
    if is_multimer: # alphafold/model/folding_multimer.py line 610.
      from alphafold.model.all_atom_multimer import get_atom37_mask
      af_a37mask = get_atom37_mask(batch['aatype']) * batch['seq_mask'][:, None]
    else:           # alphafold/model/folding.py line 509.
      af_a37mask = batch['atom37_atom_exists']
    af_aid = np.arange(np.multiply(*af_a37mask.shape)).reshape(*af_a37mask.shape)
    af_indexes = list()
    for   rid in list(self.lookups.keys()):
      for aid in list(self.lookups.get(rid, None)):
        assert not aid is None, f"Illegal `lookups[{rid}]=None`."
        assert     aid >= 0,    f"Illegal `lookups[{rid}][{aid}]<0`."
        if af_a37mask[rid][aid]==1:
          af_indexes.append(af_aid[rid][aid])
    #       print(_u.af._af_prot.residue_constants.restype_1to3[_u.af._af_prot.residue_constants.restypes[batch['aatype'][rid]]], _u.af._af_prot.residue_constants.atom_types[aid])
    # print()
    return np.asarray(af_indexes)
  
  def af_onehots(self, batch: _u.af.TAFFeatures, is_multimer: bool) -> np.ndarray:
    r"""Returns the atom one-hot flags selected from AF/AF-Multimer batch features.
    
      Args:
        batch       (TMMTop): The AF/AFMultimer batch features.
        is_multimer (bool):   If is AFMultimer.

      Returns:
        onehots (np.ndarray): The one-hot (1-selected) mask on the full indexing (N_res*37, ).
    """
    af_onehots = np.zeros((batch['aatype'].shape[0] * len(_u.af.AFAtomTypes)))
    af_onehots[self.af_indexes(batch=batch, is_multimer=is_multimer)] = 1
    return af_onehots.astype(np.int32)


class AtomIndexerByAtomType(AtomIndexer):
  r"""The atom index look-ups from residue indexes and atom type names."""

  def __init__(self, residue_indexes: list[int], atom_type_names: list[str]):
    r"""Create an atom index look-ups from residue indexes and atom type names."""
    # sanity checks.
    for rid in residue_indexes:
      assert isinstance(rid, int), f"Illegal residue index: `{rid}` of type `{type(rid)}`."
      assert rid >= 0,             f"Illegal residue index: `{rid}`."
    for atn in atom_type_names:
      assert isinstance(atn, str),     f"Illegal atom type name: `{atn}`."
      assert atn in _u.af.AFAtomTypes, f"Illegal atom type name: `{atn}`."
    # np.unique enforces sorted ordering.
    lookups = dict()
    for rid in np.unique(residue_indexes):
      lu_key = int(rid)
      lu_val = np.unique(np.asarray([_u.af.AFAtomTypeIndex.get(_, -1) for _ in atom_type_names]))
      lookups.update({lu_key: lu_val})
    AtomIndexer.__init__(self, lookups=lookups)


class AtomIndexerByCystineBond(AtomIndexerByAtomType):
  r"""The atom index look-ups from the disulfide bonds between a pair of cysteines."""

  def __init__(self, cys0: int, cys1: int):
    r"""Create an atom index look-ups from the disulfide bonds between a pair of cysteines."""
    AtomIndexerByAtomType.__init__(self, 
                                   residue_indexes=[cys0, cys1], 
                                   atom_type_names=['CB', 'SG'], )


class AtomIndexerByProlineRing(AtomIndexerByAtomType):
  r"""The atom index look-ups from the five-member ring in a proline."""

  def __init__(self, pro: int):
    r"""Create an atom index look-ups from the five-member ring in a proline."""
    AtomIndexerByAtomType.__init__(self, 
                                   residue_indexes=[pro, ], 
                                   atom_type_names=['N', 'CA', 'CB', 'CG', 'CD'], )


class AtomIndexerByPeptideBond(AtomIndexer):
  r"""The atom index look-ups from peptide bonds between adjacent residues."""

  def __init__(self, residue_index: int):
    r"""Create an atom index look-ups from peptide bonds between adjacent residues."""
    lookups = dict()
    lu_c = {residue_index  : np.asarray([_u.af.AFAtomTypeIndex.get(_, -1) for _ in ['C', 'O']])}
    lu_n = {residue_index+1: np.asarray([_u.af.AFAtomTypeIndex.get(_, -1) for _ in ['N',    ]])}
    lookups.update(lu_c)
    lookups.update(lu_n)
    AtomIndexer.__init__(self=self, lookups=lookups)


class AtomIndexerGroup:
  r"""Group of atom index look-ups for AlphaFold (N_res*37, 3) coordinates."""

  def __init__(self):
    r"""Create a group of `AtomIndexer`s."""
    self.indexers: list[AtomIndexer] = []
  
  def append(self, indexer: AtomIndexer) -> "AtomIndexerGroup":
    if isinstance(indexer, AtomIndexer): 
      self.indexers.append(indexer)
      return self
    assert False, f"Illegal `indexer` type `{type(indexer)}`."
  
  def extend(self, indexers: typing.Iterable[AtomIndexer]) -> "AtomIndexerGroup":
    for _ in indexers: 
      self = self.append(indexer=_)
    return self
  
  def __len__(self) -> int: return len(self.indexers)
  def __iter__(self) -> typing.Iterator[AtomIndexer]: return iter(self.indexers)
  def __getitem__(self, i: int) -> AtomIndexer: return self.indexers[i]
  
  def af_onehots(self, batch: _u.af.TAFResults, is_multimer: bool) -> np.ndarray:
    r"""Compute the atom indexes selected from AF/AFMultimer batch features.
    
      Args:
        batch       (TMMTop): The AF/AFMultimer batch features.
        is_multimer (bool):   If is AFMultimer.

      Returns:
        onehots (np.ndarray): The one-hot (1-selected) mask on full indexing (N_group, N_res*37).
    """
    if len(self)==0:
      return np.asarray([])
    all_indexes = tuple(i.af_onehots(batch=batch, is_multimer=is_multimer) for i in self.indexers)
    return np.vstack(all_indexes)  # (N_group, N_res*37)