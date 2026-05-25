# AlphaFold Latent Flooding: Zero-shot Prediction of Protein Conformational Ensemble

**[Zilin Song](https://github.com/ZL-Song)** (song.zilin@outlook.com)  

## Installation
If you have a working AF2 environment, install the following packages depends on your JAX/JAXLIB 
version in the environment. For more info, see the corresponding GitHub repos and find a compatible 
release.
```bash
optax
```
If installing from scratch, the following provides a non-Docker approach, depends on your cuda version (below are for CUDA 11.2):
```bash
conda create -n aftools python=3.9
conda activate aftools
conda install -y -c conda-forge openmm==7.5.1 cudatoolkit=11.2 pdbfixer
pip install -r requirements.txt --no-cache-dir
pip install --upgrade --no-cache-dir jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 optax==0.1.7 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
For unit tests, install the following:  
```bash
conda install parameterized
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

这是AlphaFold 的 Python 源代码模块。它告诉程序如何读取和处理 feature，如何构造 AlphaFold model config，如何执行 Evoformer，如何执行 Structure Module，如何输出 PDB，如何做 template / MSA pipeline，如何做 relax。 但是它不包含模型权重（例如 params_model_1_ptm.npz ），也不包含数据库（需要额外下载）。


`./test2_tem1/` The working directory for the TEM-1 example;  
`./featureizer.*` Customized scripts that executes the native AlphaFold featurization pipeline, generate `features.pkl`;  
`./test_chkp.py` Executes the AFLF checkpoint modeling for TEM-1;  
`./test2_tem1.py` Executes the AFLF inference modeling for TEM-1;  
`./prep2_tem1.py` Preprocessing file for region definitions for AFLF inference modeling for TEM-1 (requires MDAnalysis), generates `./test2_tem1/prep.pkl`;  
`./LICENSE` The AlphaFold license.

<!-- ## Reference -->
