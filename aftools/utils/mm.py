r"""AlphaFoldTools utilities for OpenMM."""
# Authors: Zilin Song.


import simtk.openmm     as mm
import simtk.openmm.app as mm_app
import simtk.unit       as mm_unit
import simtk.openmm.app.internal.pdbstructure as mm_pdbstructure


# type hints.
TMMCor = mm_unit.Quantity
TMMTop = mm_app. Topology