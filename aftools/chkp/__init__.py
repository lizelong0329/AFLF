r"""AlphaFoldTools Checkpoint."""
# Authors: Zilin Song.


from aftools.chkp._checkpoint import get_action_to
from aftools.chkp._confidence import (af_confidence, AFConfidenceModule, 
                                      af_confidence_loss, AFConfidenceLossSpecPreset, 
                                      af_multimer_confidence, AFMultimerConfidenceModule, )
from aftools.chkp import modules
from aftools.chkp import models
from aftools.chkp import runners


# NOTE (ZS):
#   checkpoint.pkl hierarchy of keys:
#     `evoformer_reprs`
#     |   `repr_msa`
#     |   `repr_pair`
#     |   `mask_msa`
#     |   `mask_pair`
#     `structmod_reprs`
#     |   `repr_single`
#     |   `repr_pair`
#     `batch`
#     |   `seq_mask`
#     |   `aatype`
#     |   `residue_index`
#     |   `atom14_atom_exists`      - AF
#     |   `atom37_atom_exists`      - AF
#     |   `residx_atom37_to_atom14` - AF
#     |   `asym_id`                 - AF-Multimer