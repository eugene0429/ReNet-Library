import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import os
import json
import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from EnvioX.TrainDataGenerator import TrainDataGenerator as DG # type: ignore
from EnvioX.TerrainGenerator import TerrainGenerator as TG # type: ignore
from EnvioX.SparseTensorProcessor import SparseTensorProcessor as SP # type: ignore
from EnvioX.Visualizer import Visualizer # type: ignore

class ReNetDataset(Dataset):

    def __init__(self, data_path, mode):
        self.inputs_path = os.path.join(data_path, 'inputs')
        self.targets_path = os.path.join(data_path, 'targets')
        self.mode = mode
    
    def __len__(self):
        return sum(1 for entry in os.scandir(self.inputs_path) if entry.is_file())
    
    def __getitem__(self, idx):

        input_path = os.path.join(self.inputs_path, f'input_{idx + 1}.json')
        target_path = os.path.join(self.targets_path, f'target_{idx + 1}.json')

        with open(input_path, 'r') as f:
            input = json.load(f)
        with open(target_path, 'r') as f:
            target = json.load(f)

        coords0_i, feats0_i = torch.tensor(input[0]), torch.tensor(input[1])
        coords0_t, feats0_t = torch.tensor(target[0][0]), torch.tensor(target[0][1])
        coords1_t, feats1_t = torch.tensor(target[1][0]), torch.tensor(target[1][1])

        coords = torch.vstack([coords0_i, coords1_t])
        coords = coords.to(dtype=torch.int)
        feats = torch.vstack([feats0_i, feats1_t])
        feats = feats.to(dtype=torch.float32)

        coords0_t, feats0_t = ME.utils.sparse_collate([coords0_t], [feats0_t])
        final_target = ME.SparseTensor(features=feats0_t, coordinates=coords0_t)
        final_target = SP.sparse_to_dense_with_size(final_target, 64)
        final_target = final_target.squeeze()

        list_of_targets = []
        for i in range(4):
            t = target[2][i]
            t = torch.tensor(t, dtype=torch.float32)
            list_of_targets.append(t)
        
        return (coords, feats), (final_target, list_of_targets)

def mean_euclidean_distance(output,
                            target
                            ):

    diff = (output - target) ** 2
    sum_diff = torch.sum(diff, dim=1)
    distance = torch.sqrt(sum_diff)
    mean_distance = torch.mean(distance)
    
    return mean_distance

def ReNet_collate_fn(batch):

    coords = []
    feats = []
    for item in batch:
        coords.append(item[0][0])
        feats.append(item[0][1])
    coords, feats = ME.utils.sparse_collate(coords, feats)
    final_targets = torch.stack([item[1][0] for item in batch])
    list_of_targets = [torch.stack([item[1][1][i] for item in batch]) for i in range(len(batch[0][1][1]))]
    
    return (coords, feats), (final_targets, list_of_targets)

def ReNet_train(model,
                mode,
                dataloader,
                optimizer,
                scheduler,
                num_epochs,
                check_progress
                ):

    print("-----------------------------------------")
    print("           ReNet train start             ")
    print("-----------------------------------------")
    
    model.train()
    
    bce_loss_fn = nn.BCELoss()
    euclidean_loss_fn = mean_euclidean_distance
    
    for epoch in range(num_epochs):

        running_loss = 0.0

        batch_index = 1

        for batch in dataloader:

            input_data, targets = batch
            coords, feats = input_data
            final_target, target_list = targets

            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

            input_data = ME.SparseTensor(features=feats, coordinates=coords, device=device)
                
            final_output, output_list = model(input_data)
            final_output = SP.sparse_to_dense_with_size(final_output, 64)
            final_output = final_output.squeeze()
                
            med_loss = euclidean_loss_fn(final_output, final_target)
                
            total_bce_loss = 0

            for output, target in zip(output_list,target_list):
                output, _, _ = output.dense()
                output = SP.dense_to_sparse(output)
                b = output.C[:, 0]
                x = output.C[:, 1]
                y = output.C[:, 2]
                z = output.C[:, 3]
                if mode == 1:
                    t = output.C[:, 4]
                    target = target[b, x, y, z, t]
                elif mode == 2:
                    target = target[b, x, y, z]

                total_bce_loss += bce_loss_fn(output.F.squeeze(), target)
            
            loss = med_loss + total_bce_loss
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                
            if check_progress:
                print(f"Epoch [{epoch+1}/{num_epochs}], batch [{batch_index}/{len(dataloader)}]")
            batch_index += 1
            
        epoch_loss = running_loss / len(dataloader)
        
        print("-----------------------------------------")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")
        print("-----------------------------------------")

        scheduler.step()

def visual_model_evaluation(model,
                            model_path,
                            test_data_path,
                            test_index
                            ):

    model.load_state_dict(torch.load(model_path))
    model.eval()

    input_path = os.path.join(test_data_path, f'inputs/input_{test_index}.json')
    target_path = os.path.join(test_data_path, f'targets/target_{test_index}.json')

    with open(input_path, 'r') as f:
        input = json.load(f)
    with open(target_path, 'r') as f:
        target = json.load(f)

    coords0_i, feats0_i = torch.tensor(input[0]), torch.tensor(input[1])
    coords0_t, _ = torch.tensor(target[0][0]), torch.tensor(target[0][1])
    coords1_t, feats1_t = torch.tensor(target[1][0]), torch.tensor(target[1][1])

    coords = torch.vstack([coords0_i, coords1_t])
    coords = coords.to(dtype=torch.int)
    feats = torch.vstack([feats0_i, feats1_t])
    feats = feats.to(dtype=torch.float32)

    coords, feats = ME.utils.sparse_collate([coords], [feats])

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    input = ME.SparseTensor(features=feats, coordinates=coords, device=device)

    with torch.no_grad():
        output, _ = model(input)

    Visualizer.visualize_sparse_tensor(output, 64)
    Visualizer.visualize_voxel(coords0_t, 64)
    
