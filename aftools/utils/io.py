r"""AlphaFoldTools utilities for I/O."""
# Authors: Zilin Song.


import os
import abc
import pickle

import numpy as np

import aftools.utils as _u


# abstract classes.
class IOWriter(abc.ABC):
  r"""The abstract persistent IO writer."""

  def __init__(self, name: str, file: str):
    r"""Constructor for the abstract persistent IO writer.
    
      Args:
        name (str): The IOWriter name.
        file (str): The IOWriter directory.
    """
    self._name = name
    self._file = file
  
  @property
  def name(self) -> str: 
    r"""The IOWriter name."""
    return self._name
  
  @property
  def file(self) -> str:
    r"""The IOWriter directory."""
    return self._file
  
  @abc.abstractmethod
  def write(self, **kwargs) -> None:
    raise NotImplementedError(r"Abstract method should be implemented by sub-classes.")
  
  @abc.abstractmethod
  def close(self) -> None:
    raise NotImplementedError(r"Abstract method should be implemented by sub-classes.")
  

class IOWriterQueue(abc.ABC):
  r"""The abstract queue of persistent IO writers."""

  def __init__(self):
    r"""Constructor for the abstract queue of persistent IO writers."""
    self.queue: list[IOWriter] = list()
    r"""The list of queued IO writers."""
  
  def close(self) -> None:
    for w in self.queue:
      w.close()
    del self


class IOStaticClass:
  r"""The IO static class that wraps the FS operations."""
  def __init__(self): raise RuntimeError("This class is static and cannot be instantiated.")
  def __new__ (self): raise RuntimeError("This class is static and cannot be instantiated.")


# dcd.
class DCDWriter(IOWriter):
  r"""The persistent DCD writer."""
  
  def __init__(self, file: str, top: _u.mm.TMMTop):
    r"""Create a persistent DCD writer.
    
      Args:
        file   (str):    The directory to which the writer outputs.
        top    (TMMTop): The OpenMM topology.
    """
    IOWriter.__init__(self, name=file, file=file)
    self._dcd = _u.mm.mm_app.DCDFile(file=open(file=self.file, mode='wb'), topology=top, dt=.1)

  def write(self, cor: _u.mm.TMMCor) -> None:
    self._dcd.writeModel(positions=cor)

  def close(self) -> None:
    self._dcd._file.close()
    del self._dcd
    del self


class DCDWriterQueue(IOWriterQueue):
  r"""The queue of persistent DCD writers."""

  def __init__(self):
    r"""Create a queue of persistent DCD writers."""
    IOWriterQueue.__init__(self)
    self.queue: list[DCDWriter]
  
  def write_mm(self, file: str, cor: _u.mm.TMMCor, top: _u.mm.TMMTop = None) -> None:
    r"""Find the writer in the queue (create new writer if not found) and write to the writer.
    
      Args:
        file   (str):    The directory to which the writer outputs.
        cor    (TMMCor): The OpenMM coordinates.
        top    (TMMTop): The OpenMM topology, default: None.
    """
    if not file in [w.name for w in self.queue]:
      assert not top is None, f"Illegal `top` is None for new DCDWriter named `{file}`."
      self.queue.append(DCDWriter(file=file, top=top))
    for w in self.queue:
      if file == w.name: w.write(cor=cor); return
    raise RuntimeError(f"`{file}` not found in DCDWriterQueue: `{[w.name for w in self.queue]}`.")
  
  def write_af(self, file: str, batch: _u.af.TAFFeatures, results: _u.af.TAFResults) -> None:
    r"""Find the writer in the queue (create new writer if not found) and write to the writer.
    
      Args:
        file    (str):         The directory to which the writer outputs.
        batch   (TAFFeatures): The AF batch features.
        results (TAFResults):  The AF output results.
    """
    pdb_string = _u.af.cast_af_prediction_to_pdb_string(batch=batch, results=results)
    pdb_struct = _u.mm.mm_pdbstructure.PdbStructure(input_stream=pdb_string.split('\n'))
    pdb_file   = _u.mm.mm_app.PDBFile(file=pdb_struct)
    self.write_mm(file=file, cor=pdb_file.getPositions(), top=pdb_file.getTopology())


# numpy.
class npy(IOStaticClass):
  r"""The IO static class that wraps the FS operations related to NumPy NPY files."""

  @staticmethod
  def dump(file: str, to_dump: np.ndarray) -> None:
    r"""Write the `to_dump` np.ndarray to `file`."""
    assert isinstance(to_dump, np.ndarray), f"Illegal `to_dump` not an `np.ndarray`."
    np.save(file=file, arr=to_dump)

  @staticmethod
  def load(file: str) -> np.ndarray:
    r"""Load the `file` as `np.ndarray`."""
    return np.load(file=file)


# pdb.
class pdb(IOStaticClass):
  r"""The IO static class that wraps the FS operations related to PDB files."""

  @staticmethod
  def dump_af(file: str, results: _u.af.TAFResults, batch: _u.af.TAFFeatures) -> None:
    r"""Write the AF prediction as PDB file.
  
      Args:
        file    (str):         The directory to which the PDB file outputs.
        results (TAFResults):  The AF output results.
        batch   (TAFFeatures): The AF batch features.
    """
    pdb_string = _u.af.cast_af_prediction_to_pdb_string(batch=batch, results=results)
    with open(file=file, mode='w') as str_file: str_file.write(pdb_string)

  @staticmethod
  def dump_mm(file: str, cor: _u.mm.TMMCor, top: _u.mm.TMMTop, ff_resnames: bool) -> None:
    r"""Write the OpenMM coordinates as PDB file.
      
      Args:
        file (str):            The directory to which the PDB file outputs.
        cor  (TMMCoordinates): The OpenMM positions.
        top  (TMMTopology):    The OpenMM topology.
        ff_resnames (bool): If use the force field residue names in the `top` for the PDB output.
    """
    # PDBFile maintains a _residueNameReplacements dict to convert the std. PDB resnames from the FF
    # resnames, e.g., HSD -> HIS. The `ff_resnames=False` replaces the resname in the topology file 
    # with the std. PDB resnames for the PDB output.
    _u.mm.mm_app.PDBFile._loadNameReplacementTables()
    if ff_resnames:
      _u.mm.mm_app.PDBFile._residueNameReplacements = {}
      _u.mm.mm_app.PDBFile.   _atomNameReplacements = {}
    with open(file=file, mode='w') as pdb_file:
      _u.mm.mm_app.PDBFile.writeModel(topology=top, positions=cor, file=pdb_file, keepIds=True)

  @staticmethod
  def load_mm(file: str) -> _u.mm.mm_app.PDBFile:
    r"""Load the `file` as `PDBFile`."""
    return _u.mm.mm_app.PDBFile(file=file)


# def _pdbline(atom_idx:  int, 
#              atom_name: str, 
#              alt_loc:   str, 
#              res_name:  str, 
#              chain_id:  str, 
#              res_idx:   int, 
#              xyz:       np.ndarray, 
#              element:   str, ) -> str:
#   r"""Build the `ATOM  ` entry line from the information given."""
#   line = ( "ATOM  "                   # Entry spec,          0- 5;
#           f"{int(atom_idx)     :>5} " # Atom index,          6-10, (11);
#           f"{str(atom_name)[:4]:<4}"  # Atom name,          12-15;
#           f"{str(alt_loc)  [ 0]   }"  # Alt. location flag, 16   ;
#           f"{str(res_name) [:4]:<4}"  # Residue name,       17-19, (20);
#           f"{str(chain_id) [:1]   }"  # Chain ID,           21;
#           f"{int(res_idx):>4}    "    # Residue index       22-25, (26-29);
#           f"{float(xyz[0]):>8.3f}"    # X,                  30-37
#           f"{float(xyz[1]):>8.3f}"    # Y,                  38-45
#           f"{float(xyz[2]):>8.3f}"    # Z,                  46-53;
#            "  1.00"                   # Occupancy,          54-59;
#            "  0.00"                   # Temperature factor, 60-65;
#            "      "                   # Empty,              66-71;
#            "    "                     # Segment ID,         72-75;
#           f"{str(element):>2}"        # Element symbol,     76-77;
#           f"  "                       # Charge,             78-19.
#            "\n")
#   return line


# pickle-able.
class pkl(IOStaticClass):
  r"""The IO static class that wraps the FS operations related to Python hashable files."""

  @staticmethod
  def dump(file: str, to_dump: object) -> None:
    r"""Write the `to_dump` hashable to `file`."""
    with open(file=file, mode='wb') as pkl_file: pickle.dump(obj=to_dump, file=pkl_file)

  @staticmethod
  def load(file: str) -> object:
    r"""Load `file` as hashable."""
    with open(file=file, mode='rb') as pkl_file: obj = pickle.load(pkl_file)
    return obj


# log.
class LOGWriter(IOWriter):
  r"""The persistent LOG writer."""

  def __init__(self, file: str):
    r"""Create a persistent LOG writer.
    
      Args:
        file   (str):  The directory to which the writer outputs.
    """
    IOWriter.__init__(self, name=file, file=file)
    self._log = open(file=file, mode='w')
  
  def write(self, log: str) -> None:
    self._log.write(log)
    self._log.flush()
  
  def close(self) -> None:
    self._log.close()
    del self


class LOGWriterQueue(IOWriterQueue):
  r"""The queue of persistent LOG writers."""

  def __init__(self):
    r"""Create a queue of persistent LOG writers."""
    IOWriterQueue.__init__(self)
    self.queue: list[LOGWriter]
  
  def write(self, file: str, log: str) -> None:
    r"""Find the writer in the queue (create new writer if not found) and write to the writer.
    
      Args:
        file   (str):  The directory to which the writer outputs.
        log    (str):  The LOG content to write.  
    """
    if not file in [w.name for w in self.queue]:
      self.queue.append(LOGWriter(file=file))
    for w in self.queue:
      if file == w.name: w.write(log=log); return
    raise RuntimeError(f"`{file}` not found in LOGWriterQueue: `{[w.name for w in self.queue]}`.")


class txt(IOStaticClass):

  @staticmethod
  def dump(file: str, to_dump: str) -> None:
    r"""Write the `to_dump` string to `file`."""
    with open(file=file, mode='w') as f: f.write(to_dump)

  @staticmethod
  def load(file: str) -> str:
    r"""Load the `file` as `str`."""
    with open(file=file, mode='r') as f: lines = f.read()
    return lines


# General IO manager.
class IOManager:
  r"""The IO manager that wraps all FS operations."""

  def __init__(self):
    r"""Create an IO manager."""
    self.dcd = DCDWriterQueue()
    self.npy = npy
    self.pdb = pdb
    self.pkl = pkl
    self.log = LOGWriterQueue()
    self.txt = txt
  
  def close(self):
    self.dcd.close()
    self.log.close()