&nbsp;

<div align="center">
<p align="center"> <img src="figure/logo.png" width="200px"> </p>


[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2406.02559)
[![NTIRE](https://img.shields.io/badge/NTIRE_Perceptual_2024-leaderboard_User🥇_ylxb-179bd3)](https://codalab.lisn.upsaclay.fr/competitions/17546#results)
[![NTIRE](https://img.shields.io/badge/NTIRE_Fidelity_2024-leaderboard_User🥈_ZXCV-179bd3)](https://codalab.lisn.upsaclay.fr/competitions/17539#results)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shadowrefiner-towards-mask-free-shadow/shadow-removal-on-adjusted-istd)](https://paperswithcode.com/sota/shadow-removal-on-adjusted-istd?p=shadowrefiner-towards-mask-free-shadow)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shadowrefiner-towards-mask-free-shadow/shadow-removal-on-istd)](https://paperswithcode.com/sota/shadow-removal-on-istd?p=shadowrefiner-towards-mask-free-shadow)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shadowrefiner-towards-mask-free-shadow/shadow-removal-on-wsrd)](https://paperswithcode.com/sota/shadow-removal-on-wsrd?p=shadowrefiner-towards-mask-free-shadow)


</div>
&nbsp;

### Introduction
This is the official PyTorch implementation of **ShadowRefiner: Towards Mask-free Shadow Removal via Fast Fourier Transformer** in CVPRW 2024. Our ShadowRefiner **won the fist place** in the [NTIRE 2024 Challenge on Shadow Removal Perceptual Track](https://codalab.lisn.upsaclay.fr/competitions/17546) and **won the second place** in the [NTIRE 2024 Challenge on Shadow Removal Fidelity Track](https://codalab.lisn.upsaclay.fr/competitions/17539).  If you find this repo useful, please give it a star ⭐ and consider citing our paper in your research. Thank you.

### Overall Framework
![Framework](figure/framework.png)

### Results
<details close>
<summary><b>Performance on ISTD, ISTD+, and WSRD+:</b></summary>

![results1](/figure/performance.png)


</details>

<details close>
<summary><b>Performance on NTIRE 2024 Shadow Removal Challenge---Perceptual Track:</b></summary>

![results1](/figure/perceptual.png)


</details>

<details close>
<summary><b>Performance on NTIRE 2024 Shadow Removal Challenge---Fidelity Track:</b></summary>

![results1](/figure/fidelity.png)


</details>




# Environment:

CUDA Version: 11.0

Python 3.8

# Dependencies:

torch==1.10.0

torchvision==0.9.0

pytorch_lightning=2.0.0

timm=0.6.12

opencv-python 

kornia

einops



# Our saved Model
Download [our saved model for NTIRE 2024 Image Shadow Removal Challenge --- Fidelity Track & Perceptual Track](https://drive.google.com/file/d/1ntXl9vGVOFGel1-Pu1vbbWidOU3QH-IM/view?usp=sharing) and unzip it into the folder ./weights to reproduce our test result.

# How to reproduce our test result
Download above saved models

Run test.py and find results in the folder ./results. Please note the weight path in Line 34 and Line 41 of test.py.

If you have any problems in reproducing our result, please contact wdong1745376@gmail.com as soon as possible.






