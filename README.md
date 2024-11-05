# ReNet Library

This project is consist of 2 parts
- ReNet
- EnvioX

ReNet is reconstruction network which is specialized for reconstructing spatial information from noisy, blinded sensor measured data. For more detailed information, please read [the paper](https://arxiv.org/abs/2206.08077).

EnvioX is library that helps to generate and process 3D point cloud & voxel data for training ReNet.

Main framework of this project is MinkowskiEngine. Please read [the document](https://nvidia.github.io/MinkowskiEngine/) for more information.

Features
-
- Custom environment point cloud data (with boxes, pillars, and walls)

- terrain point clould & voxel data **around a robot position**

- terrain point cloud & voxel data **detected by sansors on a robot** (noisy, blinded)

- point cloud & voxel data Visualizer

- ReNet training environment including ReNetDataset, ReNet_collate_fn

Requirments
-
Please refer to [MinkowskiEngine Requirements](https://github.com/NVIDIA/MinkowskiEngine/blob/master/README.md).


How to Use
-
Install `MinkowskiEngine` following the [installation guide](https://github.com/NVIDIA/MinkowskiEngine/blob/master/README.md). 

Next, install `matplotlib` & `scipy` & `numpy`

```
pip install matplotlib scipy numpy
```

Then, download ReNet & EnvioX to your project directory, and import desired features.
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from EnvioX.data_generation import DataGeneration
from EnvioX.data_processing import DataProcessing
from ReNet.model1 import ReNet as ReNet1
from ReNet.model2 import ReNet as ReNet2
from ReNet.train import train, ReNetDataset, ReNet_collate_fn

```

Please refer to Example for more detail about how to use specific functions.

