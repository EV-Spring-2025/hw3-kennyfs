# EV-HW3: PhysGaussian

This homework is based on the recent CVPR 2024 paper [PhysGaussian](https://github.com/XPandora/PhysGaussian/tree/main), which introduces a novel framework that integrates physical constraints into 3D Gaussian representations for modeling generative dynamics.

You are **not required** to implement training from scratch. Instead, your task is to set up the environment as specified in the official repository and run the simulation scripts to observe and analyze the results.

## Getting the Code from the Official PhysGaussian GitHub Repository

Download the official codebase using the following command:

```bash
git clone https://github.com/XPandora/PhysGaussian.git
```

## Environment Setup

Navigate to the "PhysGaussian" directory and follow the instructions under the "Python Environment" section in the official README to set up the environment.

On meow1 workstation of NTU CSIE, the following commands are tested working.

```bash
git clone --recurse-submodules https://github.com/XPandora/PhysGaussian
cd PhysGaussian
conda create -n PhysGaussian python=3.9
conda activate PhysGaussian
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
conda install cuda-nvcc=11.8* cuda-libraries-dev=11.8* -c nvidia
conda install gxx=11.4.0 -c conda-forge
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH
# add '#include <float.h>' to gaussian-splatting/submodules/simple-knn/simple_knn.cu manually
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/ --verbose
pip install -e gaussian-splatting/submodules/simple-knn/ --verbose
```

## Running the Simulation

Follow the "Quick Start" section and execute the simulation scripts as instructed. Make sure to verify your outputs and understand the role of physics constraints in the generated dynamics.

## Homework Instructions

Please complete Part 1–2 as described in the [Google Slides](https://docs.google.com/presentation/d/13JcQC12pI8Wb9ZuaVV400HVZr9eUeZvf7gB7Le8FRV4/edit?usp=sharing).

All videos required are included in this YouTube video:

[![Click the picture to see the video on YouTube.](https://img.youtube.com/vi/1ZgDsGhRNH8/maxresdefault.jpg)](https://www.youtube.com/watch?v=1ZgDsGhRNH8)

### Part 1

I tested `jelly` and `metal` for the model `bread_trained` with the provided config `tear_bread_config.json`. The results are included in the video.

### Part 2

The config I used are all available in [config](config/).

#### jelly

| parameter            | value | psnr  |
|----------------------|-------|-------|
| n_grid               | 10    | 20.70 |
|                      | 50    | 24.37 |
| substep_dt           | 5e-5  | 30.91 |
|                      | 2e-4  | 31.72 |
| grid_v_damping_scale | 0.9   | 21.20 |
|                      | 1.1   | 32.80 |
| softening            | 0     | inf   |
|                      | 1.0   | 65.74 |

* n_grid: When set below the default value of 150, bread sometimes fails to tear effectively. I think this is because the motion is not simulated well when the MPM background grid resolution is low.
* substep_dt: The simulation should be more realistic when this value is lower. However, I only see minor differences in shape. The limited difference may stem from the small adjustment range of the parameters (/2 and *2).
* grid_v_damping_scale: When set to 1.1, the bread rebounds significantly but doesn't explode, which I find surprising. Conversely, when set to 0.9, the bread fractures from both sides, producing small fragments. This is likely due to insufficient damping, which causes stress concentration and poor stress dispersion.
* Softening: In mpm_utils.py, we can see that this parameter only affects material 5 (plasticine). So, no difference is normal.

Insights: The n_grid parameter should be large enough to ensure an accurate simulation. The grid_v_damping_scale parameter significantly defines the object's properties, so it needs to be set carefully.

#### metal

| parameter            | value | psnr  |
|----------------------|-------|-------|
| n_grid               | 10    | 21.00 |
|                      | 50    | 26.88 |
| substep_dt           | 5e-5  | 37.36 |
|                      | 2e-4  | 39.31 |
| grid_v_damping_scale | 0.9   | 21.96 |
|                      | 1.1   | 28.41 |
| softening            | 0     | 82.68 |
|                      | 1.0   | 82.97 |

The observation and PSNR mostly match jelly's.

### Bonus

Make the material-related parameters differentiable. Then, optimize them as we did in HW1 (3D Gaussian Splatting). However, this will obviously be very time-consuming.

## Reference

```bibtex
@inproceedings{xie2024physgaussian,
    title     = {Physgaussian: Physics-integrated 3d gaussians for generative dynamics},
    author    = {Xie, Tianyi and Zong, Zeshun and Qiu, Yuxing and Li, Xuan and Feng, Yutao and Yang, Yin and Jiang, Chenfanfu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2024}
}
```
