import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import os
import json
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from torch.utils.data import Dataset
from EnvioX.TerrainGenerator import TerrainGenerator as TG # type: ignore
from EnvioX.SparseTensorProcessor import SparseTensorProcessor as SP # type: ignore

class ReNetDataset(Dataset):

    def __init__(self, data_path, model_mode, data_type):
        self.inputs_path = os.path.join(data_path, 'inputs')
        self.targets_path = os.path.join(data_path, 'targets')
        self.mode = model_mode
        self.type = data_type
    
    def __len__(self):
        return sum(1 for entry in os.scandir(self.inputs_path) if entry.is_file())
    
    def __getitem__(self, idx):

        input_path = os.path.join(self.inputs_path, f'input_{idx + 1}.json')
        target_path = os.path.join(self.targets_path, f'target_{idx + 1}.json')

        with open(input_path, 'r') as f:
            input = json.load(f)
        with open(target_path, 'r') as f:
            target = json.load(f)

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        coords0_i, feats0_i = torch.tensor(input[0]), torch.tensor(input[1])
        coords0_t, feats0_t = torch.tensor(target[0][0]), torch.tensor(target[0][1])
        coords1_t, feats1_t = torch.tensor(target[1][0]), torch.tensor(target[1][1])

        coords = torch.vstack([coords0_i, coords1_t])
        coords = coords.to(torch.int).to(device)
        feats = torch.vstack([feats0_i, feats1_t])
        feats = feats.to(torch.float32).to(device)

        coords0_t, feats0_t = ME.utils.sparse_collate([coords0_t], [feats0_t])
        final_target = ME.SparseTensor(features=feats0_t.to(device),
                                       coordinates=coords0_t.to(device),
                                       device=device
                                       )
        final_target = SP.sparse_to_dense_with_size(final_target, 64)
        final_target = final_target.squeeze()

        if self.type == 'train':
            list_of_targets = []
            for i in range(4):
                t = target[2][i]
                t = torch.tensor(t, dtype=torch.float32, device=device)
                list_of_targets.append(t)
            
            return (coords, feats), (final_target, list_of_targets)
        
        elif self.type == 'val':
            return (coords, feats), (coords0_t, feats0_t)


def mean_euclidean_distance(output, target):

    diff = (output - target) ** 2
    sum_diff = torch.sum(diff, dim=1)
    distance = torch.sqrt(sum_diff)
    mean_distance = torch.mean(distance)
    
    return mean_distance

def ReNet_collate_fn_train(batch):

    coords = []
    feats = []
    for item in batch:
        coords.append(item[0][0])
        feats.append(item[0][1])
    coords, feats = ME.utils.sparse_collate(coords, feats)
    final_targets = torch.stack([item[1][0] for item in batch])
    list_of_targets = [torch.stack([item[1][1][i] for item in batch]) for i in range(len(batch[0][1][1]))]
    
    return (coords, feats), (final_targets, list_of_targets)

def ReNet_collate_fn_val(batch):

    coords = []
    feats = []
    coords_t = []
    feats_t = []
    for item in batch:
        coords.append(item[0][0])
        feats.append(item[0][1])
        coords_t.append(item[1][0])
        feats_t.append(item[1][1])
    coords, feats = ME.utils.sparse_collate(coords, feats)
    coords_t, feats_t = ME.utils.sparse_collate(coords_t, feats_t)
    
    return (coords, feats), (coords_t, feats_t)

def ReNet_train(model,
                model_mode,
                dataloader,
                optimizer,
                scheduler,
                num_epochs,
                model_path,
                check_progress
                ):
    
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

            model = model.to(device)

            input_data = ME.SparseTensor(features=feats.to(device),
                                         coordinates=coords.to(device),
                                         device=device
                                         )
            
            final_output, output_list = model(input_data)
            final_output = SP.sparse_to_dense_with_size(final_output, 64)
            final_output = final_output.squeeze()
                
            med_loss = euclidean_loss_fn(final_output, final_target)
                
            total_bce_loss = 0

            for output, target in zip(output_list,target_list):
                output, _, _ = output.dense()
                output = SP.dense_to_sparse(output)
                b = output.C[:, 0].long()
                x = output.C[:, 1].long()
                y = output.C[:, 2].long()
                z = output.C[:, 3].long()
                if model_mode == 1:
                    t = output.C[:, 4].long()
                    target = target[b, x, y, z, t]
                elif model_mode == 2:
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

        model_path_ = os.path.join(model_path, f"ReNet_{epoch_loss}.pth")
        torch.save_state_dict(model, model_path_)
        print(f"Model parameters saved to {model_path_}")
        
        print("-----------------------------------------")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")
        print("-----------------------------------------")

        scheduler.step()

def ReNet_validation(model,
                     model_path,
                     dataloader
                     ):

    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    TP = 0
    FP = 0
    FN = 0

    with torch.no_grad():

        for batch in dataloader:

            input_data, final_target = batch
            coords_t, _ = final_target
            coords_t = coords_t.long()
            coords, feats = input_data

            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

            model = model.to(device)

            input_data = ME.SparseTensor(features=feats,
                                         coordinates=coords,
                                         device=device
                                         )
            
            final_output, _ = model(input_data)
            coords_o = final_output.C.long()

            b = dataloader.batch_size

            out = torch.zeros([b, 64, 64, 64])
            out[coords_o[:, 0], coords_o[:, 1], coords_o[:, 2], coords_o[:, 3]] = 1

            tar = torch.zeros([b, 64, 64, 64])
            tar[coords_t[:, 0], coords_t[:, 1], coords_t[:, 2], coords_t[:, 3]] = 1

            tar_flat = tar.view(-1)
            out_flat = out.view(-1)

            TP += torch.sum((tar_flat == 1) & (out_flat == 1)).item()
            FP += torch.sum((tar_flat == 0) & (out_flat == 1)).item()
            FN += torch.sum((tar_flat == 1) & (out_flat == 0)).item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1_score: {f1_score}')

    