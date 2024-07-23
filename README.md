# CAAE-PCL-MIMR
The source code of the Integration of Physics-Constrained Learning with Adversarial Autoencoders for Simultaneous Inference of Hydraulic Conductivity and Contaminant Source.

# The directory structure

```
  CAAE-PCL-MIMR                           
  ├── CVAEPCL        % This dir stores the main codes
    ├──dataset       % This dir stores the codes for the log-conductivity field dataset
    ├──MAIN          % This dir stores the CAAE-PCL codes run on the candidate observation wells, as well as the CAAE codes 
    ├──MIMR          % This dir stores the codes for the observation well optimization strategy 
    └──MIMR-MAIN     % This dir stores the CAAE-PCL codes run on selected observation wells 
  ├── GroundwaterModelPytorch      % This dir stores the customized PDE solver layer (FDM)
  ├── sparse_linear_systems        % This dir stores the large-scale linear algebra solver
  └── Utils           % This dir stores the visualizing codes
```


# Installation
~~~
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install h5py
pip install scipy
pip install matplotlib
pip install torch_sparse_solve
pip install cupy-cuda12x
python -m cupyx.tools.install_library --cuda 11.x --library cutensor
~~~

# Getting Started

## Test the results in the paper
Run the following source code:
~~~
python ./CVAEPCL/MAIN/plot_step1.py
python ./CVAEPCL/MAIN/plot_step2.py
python ./CVAEPCL/MIMR/plot_MIMR.py
python ./CVAEPCL/MAIN-MIMR/plot_step1.py
python ./CVAEPCL/MAIN-MIMR/plot_step2.py
~~~

## Train the model yourself and generate results
Run the following code in sequence:
~~~
python ./CVAEPCL/dataset/make_K_dataset.py
python ./CVAEPCL/MAIN/train_gan.py
python ./CVAEPCL/MAIN/train_step1.py
python ./CVAEPCL/MAIN/train_step2.py
python ./CVAEPCL/MIMR/monte_carlo_simulation.py
python ./CVAEPCL/MIMR/MIMR_run.py
python ./CVAEPCL/MAIN-MIMR/train_step1.py
python ./CVAEPCL/MAIN-MIMR/train_step2.py
~~~


