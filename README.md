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

    <img width="300" alt="스크린샷 2024-11-09 오전 12 06 13" src="https://github.com/user-attachments/assets/c4d13901-5109-48cb-8d8b-51adbaff49e0">

- terrain point cloud & voxel data **detected by sansors on a robot** (noisy, blinded)

  <img width="300" alt="스크린샷 2024-11-09 오전 12 07 03" src="https://github.com/user-attachments/assets/f4d75f0e-f2e0-4940-98ee-747c740903c1"> <img width="290" alt="스크린샷 2024-11-09 오전 12 08 11" src="https://github.com/user-attachments/assets/e92daf55-d047-46b1-9e6f-4373c2f584f1">

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

