import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from ReNet.model1 import ReNet as ReNet1 # type: ignore
from ReNet.model2 import ReNet as ReNet2 # type: ignore
from ReNet.TrainHelper import ReNetDataset, ReNet_collate_fn_train, ReNet_train # type: ignore

model_mode = 1

if model_mode == 1:
    model = ReNet1(in_channels=3,
                   out_channels=3,
                   D=4,
                   alpha=0.5
                   )
elif model_mode == 2:
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

model_path = ""
train_data_path = "data/train"

def main():
    dataset = ReNetDataset(train_data_path,
                           model_mode=model_mode,
                           data_type='train'
                           )

    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=ReNet_collate_fn_train
                            )
    
    print("-----------------------------------------")
    print("           ReNet train start             ")
    print("-----------------------------------------")

    ReNet_train(model=model,
                model_mode=model_mode,
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=num_epochs,
                model_path=model_path,
                check_progress=True
                )    

if __name__ == "__main__":
    main()