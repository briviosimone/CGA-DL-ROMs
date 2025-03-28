# CGA-DL-ROM

This repository contains the source code implementation (with one example) of the manuscript: 

*S. Brivio, S. Fresca, A. Manzoni, [Handling geometrical variability in nonlinear reduced order modeling through Continuous Geometry-Aware DL-ROMs](https://arxiv.org/abs/2411.05486v1) (2024).*

### Main dependencies
We recommend to install the code library in a new conda environment:
```
conda create -n cga python=3.10
conda activate cga
conda install -c conda-forge cudatoolkit=11.8.0
conda install -c nvidia cuda-nvcc
pip install -r requirements.txt --no-cache-dir
bash cuda-init.sh
```

### Instructions
For the hyperelasticity example, 

1. From the main folder, generate the data folder ```./data``` with ```mkdir data```. 
2. Download in ```./data``` the dataset from the [drive](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) provided by the [Geo-FNO](https://github.com/neuraloperator/Geo-FNO?tab=readme-ov-file) authors.
3. From the main folder, run ```python preprocess.py train``` to preprocess the datasets.
4. Go to ```./test_cases/``` and then use ```python elasticity.py train``` to run both training and testing phases or ```python elasticity.py test``` to run only the testing phase.

The results of the experiment are then available in the folder ```./results/elasticity/```.

### Cite
If the present repository and/or the original paper was useful in your research, 
please consider citing

```
@misc{brivio2024cga,
      title={Handling geometrical variability in nonlinear reduced order modeling through Continuous Geometry-Aware DL-ROMs}, 
      author={Simone Brivio and Stefania Fresca and Andrea Manzoni},
      year={2024},
      eprint={2411.05486},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2411.05486}, 
}
```

