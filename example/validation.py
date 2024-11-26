import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from torch.utils.data import DataLoader
from ReNet.model1 import ReNet as ReNet1 # type: ignore
from ReNet.model2 import ReNet as ReNet2 # type: ignore
from ReNet.TrainHelper import ReNetDataset, ReNet_collate_fn_val, ReNet_validation # type: ignore

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

batch_size = 2
val_data_path = "data/val"
model_path = "best_ReNet.pth"

def main():
    dataset = ReNetDataset(val_data_path,
                           model_mode=model_mode,
                           data_type='val'
                           )

    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=ReNet_collate_fn_val
                            )

    ReNet_validation(model=model,
                     model_path=model_path,
                     dataloader=dataloader
                     )
    
if __name__ == "__main__":
    main()