# AdaPool: Exponential Adaptive Pooling for Information-Retaining Downsampling
![supported versions](https://img.shields.io/badge/python-3.x-brightgreen/?style=flat&logo=python&color=green)
![Library](https://img.shields.io/badge/library-PyTorch-blue?logo=Pytorch)
![GitHub license](https://img.shields.io/cocoapods/l/AFNetworking)


--------------------------------------------------------------------------------
## Abstract
Pooling layers are essential building blocks of Convolutional Neural Networks (CNNs) that reduce computational overhead and increase the receptive fields of proceeding convolutional operations. They aim to produce downsampled volumes that closely resemble the input volume while, ideally, also being computationally and memory efficient. It is a challenge to meet both requirements jointly. To this end, we propose an adaptive and exponentially weighted pooling method named <i>adaPool</i>. Our proposed method uses a parameterized fusion of two sets of pooling kernels that are based on the exponent of the Dice-Sørensen coefficient and the exponential maximum, respectively. A key property of adaPool is its bidirectional nature. In contrast to common pooling methods, weights can be used to upsample a downsampled activation map. We term this method <i>adaUnPool</i>. We demonstrate how adaPool improves the preservation of detail through a range of tasks including image and video classification and object detection. We then evaluate adaUnPool on image and video frame super-resolution and frame interpolation tasks. For benchmarking, we introduce <i>Inter4K</i>, a novel high-quality, high frame-rate video dataset. Our combined experiments demonstrate that adaPool systematically achieves better results across tasks and backbone architectures, while introducing a minor additional computational and memory overhead. <p align="center">

<p align="center">
<img src="./images/adaPool_cover.png" width="700" />
</p>

<i></i>
<br>
<p align="center">
<a href="#" target="blank" >[arXiv preprint -- coming soon]</a>
</p>


## Dependencies
All parts of the code assume that `torch` is of version 1.4 or higher. There might be instability issues on previous versions.

This work relies on the previous repo for exponential maximum pooling (**[alexandrosstergiou/SoftPool](https://github.com/alexandrosstergiou/SoftPool)**). Before opening an issue please do have a look at that repository as common problems in running or installation have been addressed.

> ***! Disclaimer:*** This repository is heavily structurally influenced on Ziteng Gao's LIP repo [https://github.com/sebgao/LIP](https://github.com/sebgao/LIP)

## Installation

You can build the repo through the following commands:
```
$ git clone https://github.com/alexandrosstergiou/adaPool.git
$ cd adaPool-master/pytorch
$ make install
--- (optional) ---
$ make test
```


## Usage

You can load any of the 1D, 2D or 3D variants after the installation with:

```python
# Ensure that you import `torch` first!
import torch
import adapool_cuda

# For function calls
from adaPool import adapool1d, adapool2d, adapool3d, adaunpool
from adaPool import edscwpool1d, edscwpool2d, edscwpool3d
from adaPool import empool1d, empool2d, empool3d
from adaPool import idwpool1d, idwpool2d, idwpool3d

# For class calls
from adaPool import AdaPool1d, AdaPool2d, AdaPool3d
from adaPool import EDSCWPool1d, EDSCWPool2d, EDSCWPool3d
from adaPool import EMPool1d, EMPool2d, EMPool3d
from adaPool import IDWPool1d, IDWPool2d, IDWPool3d
```

+ `(ada/edscw/em/idw)pool<x>d`: Are functional interfaces for each of the respective pooling methods.
+ `(Ada/Edscw/Em/Idw)Pool<x>d`: Are the class version to create objects that can be referenced in the code.

## Citation

```
@article{stergiou2021adapool,
  title={AdaPool: Exponential Adaptive Pooling for Information-Retaining Downsampling},
  author={Stergiou, Alexandros and Poppe, Ronald},
  journal={arXiv preprint},
  year={2021}}
```

## Licence

MIT
