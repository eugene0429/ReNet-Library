# ReNet Library

This project is consist of 2 parts
- ReNet
- EnvioX

ReNet is Reconstruction Network which is specialized for reconstructing spatial information from noisy, blinded sensor measured data. For more detailed information, please read [the paper](https://arxiv.org/abs/2206.08077).

EnvioX is library that helps to generate and process 3D point cloud & voxel data for training ReNet.

Main framework of this project is MinkowskiEngine. Please read [the document](https://nvidia.github.io/MinkowskiEngine/) for more information.

Features
-
- Custom environment point cloud data (with boxes, pillars, and walls)

- terrain point clould & voxel data **around a robot position**

    <img width="300" alt="Figure_1" src="https://github.com/user-attachments/assets/6038ae1b-c14e-4307-9261-aa9f03c59ab9"> <img width="300" alt="Figure_2" src="https://github.com/user-attachments/assets/72990891-9838-4e24-9c3b-a00e95766476">

- terrain point cloud & voxel data **detected by sansors on a robot** (noisy, blinded)

  <img width="300" alt="Figure_3" src="https://github.com/user-attachments/assets/1f20e9a5-6d32-4d13-a8d4-d04f4beb172f"> <img width="300" alt="Figure_4" src="https://github.com/user-attachments/assets/9ad30a11-eb5e-43d7-8724-370dc0159ce3">

- point cloud & voxel data Visualizer

- ReNet training environment including ReNetDataset, ReNet_collate_fn

Requirments
-
Please refer to [MinkowskiEngine Requirements](https://github.com/NVIDIA/MinkowskiEngine/blob/master/README.md#requirements).


How to Use
-
Install `MinkowskiEngine` following the [installation guide](https://github.com/NVIDIA/MinkowskiEngine/blob/master/README.md#installation). 

Next, install `matplotlib` & `scipy` & `numpy`

```
pip install matplotlib scipy numpy
```

Then, download ReNet & EnvioX to your project directory, and import desired features.
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from EnvioX.TrainDataGenerator import generate_dataset
from EnvioX.TerrainGenerator import TerrainGenerator as TG
from EnvioX.SparseTensorProcessor import SparseTensorProcessor as SP
from EnvioX.Visualizer import Visualizer
from ReNet.model1 import ReNet as ReNet1
from ReNet.model2 import ReNet as ReNet2
from ReNet.TrainHelper import ReNetDataset, ReNet_collate_fn_train, ReNet_train

```

Please refer to Example for more detail about how to use specific functions.

For Colab user
-

Refer to [ReNet.ipynb](https://colab.research.google.com/drive/15NKy6zd6vTYhXXXuDpYYWh2dAgMrxGkp?usp=sharing). This notebook file contains from setting environments for `MinkowskiEngine` to data generation, train, validation and test.

