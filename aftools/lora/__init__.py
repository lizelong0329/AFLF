r"""AlphaFoldTools Low-Rank Adaptation (LoRA)."""
# Authors: Zilin Song.


from aftools.lora._lora import apply_func, lora_checkpoint, LoraCheckpointPreset
from aftools.lora import models


# NOTE (ZS):
#   lora_checkpoint.pkl hierarchy of keys:
#     `evoformer_lora_feats`
#     |   `lora_feat_msa`  - `A`, `B`, `C`
#     |   `lora_feat_pair` - `A`, `B`, `C`
#     `structmod_lora_feats`
#     |   `lora_feat_single` - `A`, `B`
#     |   `lora_feat_pair`   - `A`, `B`, `C`
#     `evoformer_lora_masks`
#     |   `lora_mask_msa`
#     |   `lora_mask_pair`
#     `structmod_lora_masks`
#     |   `lora_mask_single`
#     |   `lora_mask_pair`
#     `evoformer_reprs` - From checkpoint.pkl
#     |   `repr_msa`
#     |   `repr_pair`
#     |   `mask_msa`
#     |   `mask_pair`
#     `structmod_reprs` - From checkpoint.pkl
#     |   `repr_single`
#     |   `repr_pair`
#     `batch`           - From checkpoint.pkl
#     |   `seq_mask`
#     |   `aatype`
#     |   `residue_index`
#     |   `atom14_atom_exists`      - AF
#     |   `atom37_atom_exists`      - AF
#     |   `residx_atom37_to_atom14` - AF
#     |   `asym_id`                 - AF-Multimer