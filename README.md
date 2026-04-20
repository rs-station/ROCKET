# *R*efining *O*penfold predictions with *C*rystallographic/*C*ryo-EM Li*KE*lihood *T*argets (ROCKET)

![PyPI - Version](https://img.shields.io/pypi/v/rs-rocket)
[![Build](https://github.com/alisiafadini/ROCKET/actions/workflows/test.yml/badge.svg)](https://github.com/alisiafadini/ROCKET/actions/workflows/test.yml)
[![Ruff](https://github.com/alisiafadini/ROCKET/actions/workflows/lint.yml/badge.svg)](https://github.com/alisiafadini/ROCKET/actions/workflows/lint.yml)
[![GitHub License](https://img.shields.io/github/license/alisiafadini/ROCKET)](https://github.com/alisiafadini/ROCKET/blob/main/LICENSE)
[![Nature Methods](https://img.shields.io/badge/DOI-10.1038%2Fs41592--026--03047--4-purple?link=https%3A%2F%2Fwww.nature.com%2Farticles%2Fs41592-026-03047-4)](https://www.nature.com/articles/s41592-026-03047-4)
[![Doc](https://img.shields.io/badge/Doc-GitBook-violet)](https://rocket-9.gitbook.io/rocket-docs)




This is the code repo for [AlphaFold as a Prior: Experimental Structure Determination Conditioned on a Pretrained Neural Network](https://www.nature.com/articles/s41592-026-03047-4)

You can find detailed documentation and walk-through tutorials in [our GitBook](https://rocket-9.gitbook.io/rocket-docs) and our [SBGrid webinar](https://www.youtube.com/watch?v=_29CpGPqIQA).

## Installation

<details>
<summary><span style="font-size:1.3em;font-weight:bold;">1. Install OpenFold</span></summary>

To ensure usability, we forked the OpenFold repo, and sorted a couple details in the installation guides. Here is what we advise ROCKET users to do:

**Note**: The ⁠openfold installation requires approximately 6 GB of free space to download weights. Please ensure you start in a directory with sufficient available space.

**Note**: To ensure a smooth installation and execution of ROCKET, install on a GPU machine that matches the hardware you’ll use in production. In other words, for HPC users, if you plan to run your code on a node with a particular GPU model, request the same GPU model when you install OpenFold. This is important because the installation process performs a hardware-specific compilation. We also recommend using GPUs with CUDA Compute Capability 8.0 or higher.

1. Clone our fork of the OpenFold repo, switch to the `pl_upgrades` branch to work with CUDA 12:

    ```
    git clone https://github.com/minhuanli/rocket_openfold.git
    cd rocket_openfold
    git checkout pl_upgrades
    ```

2. Create a conda/mamba env with the `environment.yml`
   
   
    Note: If you work with an HPC cluster with package management like `module`, purge all your modules before this step to avoid conflicts. 
    
    ```
    mamba env create -n <env_name_you_like> -f environment.yml
    mamba activate <env_name_you_like>
    ```
 
    The main change we made is moving the `flash-attn` package outside of the yml file, so you can install it manually afterwards. This is necessary because this OpenFold version relies on pytorch 2.1, which is incompatible with the latest flash-attn, so a simple `pip install flash-attn` would fail. Also using a `--no-build-isolation` flag allows using `ninja` for compilation, which is much faster.
 
   


3. Install compatible `flash-attn` (latest flash-attn with noted support for pytroch-2.1 + cuda-12.1)

    ```
    pip install flash-attn==2.2.2 --no-build-isolation
    ```

4. Run the setup script to install OpenFold, and configure kernels and folding resources
   
    ```
    ./scripts/install_third_party_dependencies.sh
    ```
 
    Add the following lines to `<path_to_your_conda_env>/etc/conda/activate.d/env_vars.sh`, create it if it doesn't exist
    
    ```
    #!/bin/sh
    
    export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    ```
 
    This is so everytime you activate this env, the prepend will happen automatically.

5. Download AlphaFold2 weights, add **the resources path to system environment** (we need this for ROCKET)
   
    ```
    ./scripts/download_alphafold_params.sh ./openfold/resources
    ```
 
    Note: You can download OpenFold weights if you want to try

    Append the following line to `<path_to_your_conda_env>/etc/conda/activate.d/env_vars.sh`, you should have created it from the previous step

    ```
    export OPENFOLD_RESOURCES="<ABSOLUTE_PATH_TO_OPENFOLD_FOLDER>/openfold/resources"
    ```

    `<ABSOLUTE_PATH_TO_OPENFOLD_FOLDER>` should be the output of `pwd -P` you get from the OpenFold repo path.

    Deactivate and reactivate your python environment, you should be able to run and see the path:
    
    ```
    echo $OPENFOLD_RESOURCES 
    ```

6. Check your OpenFold build with unit tests:

    ```
    ./scripts/run_unit_tests.sh
    ```
 
    Ensure you see no errors:
    
    ```
    ...
    Time to load evoformer_attn op: 243.8257336616516 seconds
    ............s...s.sss.ss.....sssssssss.sss....ssssss..s.s.s.ss.s......s.s..ss...ss.s.s....s........
    ----------------------------------------------------------------------
    Ran 117 tests in 275.889s
 
    OK (skipped=41)
    ```   

</details>



<details>
<summary><span style="font-size: 1.3em; font-weight: bold;">2. Install Phenix (required for automatic preprocessing and post-refinement)</span></summary>

[Phenix](https://phenix-online.org/) is required for automatic data preprocessing and for post-refinement when polishing final model geometry. Follow the steps below to install it and **add the path to the system environment variables**:

1. Download the latest `nightly-build` Phenix python3 installer according to [https://phenix-online.org/download](https://phenix-online.org/download), note you have download the installer from the [show-all link](https://www.phenix-online.org/download/nightly_builds.cgi?show_all=1), with version newer than `2.0rc1-5647`

2. Run the installer

    ```
    bash phenix-installer-2.0rc1-5617-<platform>.sh
    ```

    You will be prompted to type your preferred path of installation, after specifying it, you will see:

    ```
    Phenix will now be installed into this location:
    <phenix_directory>/phenix-2.0rc1-5617
    ```

    Note: `<phenix_directory>` must be a absolute path. The installer will will make `<phenix_directory>/phenix-2.0rc1-5617` and install there.

3. Append the following line to `<path_to_your_conda_env>/etc/conda/activate.d/env_vars.sh`, you should have created it from the previous section

    ```
    export PHENIX_ROOT="<phenix_directory>/phenix-2.0rc1-5617"
    ```

    `<phenix_directory>` is where you install Phenix in the last step

    Deactivate and reactivate your python environment, you should be able to run and see the path:
    
    ```
    echo $PHENIX_ROOT 
    ``` 

</details>



### 3. Install ROCKET

ROCKET is available through pypi as `rs-rocket`, you can easily install by run

```
pip install rs-rocket
```

For developer mode, you can fetch the repo and do editable installation locally 

```
git clone https://github.com/rs-station/ROCKET.git
cd ROCKET
pip install -e ".[tests,CI]"
```


Run `rk.score --help` after installation, if you see a normal doc strings without errors, you are good to go!


### Citing

```
@article{fadini2026,
  title={AlphaFold as a prior: experimental structure determination conditioned on a pretrained neural network},
  author={Fadini, Alisia and Li, Minhuan and McCoy, Airlie J. and Banjara, Suresh and Okumura, Hiroki and Napier, Eve and Fontana, Pietro and Khan, Amir R. and Jovine, Luca and Terwilliger, Thomas C. and Read, Randy J. and Hekstra, Doeke R. and AlQuraishi, Mohammed},
  journal={Nature Methods},
  volume={23},
  pages={785--795},
  year={2026},
  doi={10.1038/s41592-026-03047-4}
}
```

   
