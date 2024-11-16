import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from EnvioX.TrainDataGenerator import TrainDataGenerator as DG # type: ignore
from ReNet.model1 import ReNet as ReNet1 # type: ignore
from ReNet.model2 import ReNet as ReNet2 # type: ignore
from ReNet.train import ReNetDataset, ReNet_collate_fn, ReNet_train # type: ignore
from ReNet.train import visual_model_evaluation # type: ignore

mode = 1

if mode == 1:
    model = ReNet1(in_channels=3,
                   out_channels=3,
                   D=4,
                   alpha=0.5
                   )
elif mode == 2:
    model = ReNet2(in_channels=3,
                   out_channels=3,
                   D=4,
                   alpha=0.5
                   )

num_epochs = 2
batch_size = 2

lr_i = 0.01
lr_f = 0.0001

optimizer = optim.Adam(model.parameters(), lr=lr_i)

gamma = (lr_f / lr_i) ** (1 / num_epochs)  
scheduler = ExponentialLR(optimizer, gamma=gamma)

save_model = True
model_path = ""

generate_new_data = True
data_path = "data"
os.makedirs(data_path, exist_ok=True)

train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')
os.makedirs(train_data_path, exist_ok=True)
os.makedirs(test_data_path, exist_ok=True)

sensor_config = {'tilt_angle': 30,
                 'fov_angle': 80, 
                 'detection_distance': 2,
                 'relative_position': {'front': [0.5, 0.0, 0.0], 
                                       'back': [-0.5, 0.0, 0.0], 
                                       'right': [0.0, -0.2, 0.0], 
                                       'left': [0.0, 0.2, 0.0]}
                 }

if generate_new_data:
    DG.generate_dataset(grid_size=20,
                        detection_range=3.2,
                        robot_size=[0.4, 1.0, 0.8],
                        robot_speed=1.0,
                        sensor_config=sensor_config,
                        point_density=15,
                        num_env_configs=2,
                        num_data_per_env=3,
                        num_time_step=2,
                        time_step=0.2,
                        save_path=train_data_path,
                        mode=mode
                        )

    DG.generate_dataset(grid_size=20,
                        detection_range=3.2,
                        robot_size=[0.4, 1.0, 0.8],
                        robot_speed=1.0,
                        sensor_config=sensor_config,
                        point_density=15,
                        num_env_configs=1,
                        num_data_per_env=2,
                        num_time_step=2,
                        time_step=0.2,
                        save_path=test_data_path,
                        mode='test'
                        )
    print("-----------------------------------------")
    print("           Data is generated             ")
    print("-----------------------------------------")

dataset = ReNetDataset(train_data_path, mode=mode)

dataloader = DataLoader(dataset=dataset, 
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=ReNet_collate_fn
                        )

ReNet_train(model=model,
            mode=mode,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            check_progress=True
            )

if save_model:
    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to {model_path}")

visual_model_evaluation(model=model,
                        model_path=model_path,
                        test_data_path=test_data_path,
                        test_index=2
                        )