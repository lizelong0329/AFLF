# AlphaFold Latent Flooding: Zero-shot generation of conformational ensembles.

**[Zilin Song](https://github.com/ZL-Song)** (song.zilin@outlook.com)  

## Installation
If you have a working AF2 environment, install the following packages depends on your JAX/JAXLIB 
version in the environment. For more info, see the corresponding GitHub repos and find a compatible 
release.
```bash
chex
optax
```
If installing from scratch, the following provides a non-Docker approach, depends on your CUDA version (below are for CUDA 11):
```bash
conda create --name aftools python==3.9
conda activate aftools
conda install -y -c conda-forge openmm==7.5.1 cudatoolkit=11.1.1 pdbfixer
pip install -r requirements.txt --no-cache-dir
pip install --upgrade --no-cache-dir jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 optax==0.1.7 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install -c nvidia cuda-nvcc
```

## Example with TEM-1
For AFLF checkpoint modeling:
```bash
python test_chkp.py
```
For AFLF inference modeling:
```bash
python test2_tem1.py
```

## File descriptions
`./afmisc/` Miscellaneous files from AlphaFold repository;  
`./aftools/` Implements the AFLF algorithm and helper functions;  
`./alphafold/` The AlphaFold source code;  
`./test2_tem1/` The working directory for the TEM-1 example;  
`./featureizer.*` Customized scripts that executes the native AlphaFold featurization pipeline, generate `features.pkl`;  
`./test_chkp.py` Executes the AFLF checkpoint modeling for TEM-1;  
`./test2_tem1.py` Executes the AFLF inference modeling for TEM-1;  
`./prep2_tem1.py` Preprocessing file for region definitions for AFLF inference modeling for TEM-1 (requires MDAnalysis), generates `./test2_tem1/prep.pkl`;  
`./LICENSE` The AlphaFold license.

<!-- ## Reference -->