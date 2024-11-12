import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from EnvioX.data_generation import DataGeneration # type: ignore
from EnvioX.data_processing import DataProcessing # type: ignore
from ReNet.model1 import ReNet as ReNet1 # type: ignore
from ReNet.model2 import ReNet as ReNet2 # type: ignore
from ReNet.train import ReNetDataset, ReNet_collate_fn, ReNet_train # type: ignore

DG = DataGeneration()
DP = DataProcessing()

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

save_model = False
save_path = "ReNet_parameters.pth"

sensors_config = {'tilt_angle': 30,
                 'fov_angle': 80, 
                 'detection_distance': 2,
                 'relative_position': {'front': [0.5, 0.0, 0.0], 
                                       'back': [-0.5, 0.0, 0.0], 
                                       'right': [0.0, -0.2, 0.0], 
                                       'left': [0.0, 0.2, 0.0]}
                 }

input_lists, target_lists = DG.generate_dataset(grid_size=20,
                                                detection_range=3.2,
                                                robot_size=[0.4, 1.0, 0.8],
                                                robot_speed=1.0,
                                                sensors_config=sensors_config,
                                                point_density=15,
                                                num_env_configs=2,
                                                num_data_per_env=2,
                                                time_step=0.1,
                                                num_time_step=2,
                                                visualize=False
                                                )
print("Data is generated")

dataloaders = []
for i in range(1, len(input_lists) + 1):
    dataset = ReNetDataset(input_lists[f'list_{i}'], 
                           target_lists[f'list_{i}'],
                           mode=mode
                           )
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=ReNet_collate_fn
                            )
    dataloaders.append(dataloader)

print("Dataloaders are set")

ReNet_train(model=model,
            mode=mode,
            dataloaders=dataloaders,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            check_progress=True
            )

if save_model:
    torch.save(model.state_dict(), save_path)
    print(f"Model parameters saved to {save_path}")