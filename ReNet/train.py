import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from EnvioX.TrainDataGenerator import TrainDataGenerator as DG # type: ignore
from EnvioX.TerrainGenerator import TerrainGenerator as TG # type: ignore
from EnvioX.SparseTensorProcessor import SparseTensorProcessor as SP # type: ignore

class ReNetDataset(Dataset):

    def __init__(self, input_data_list, targets_list, mode):
        self.input_data_list = input_data_list
        self.targets_list = targets_list
        self.mode = mode
    
    def __len__(self):
        return len(self.input_data_list)
    
    def __getitem__(self, idx):

        input0 = self.input_data_list[idx][1]
        input1 = self.input_data_list[idx][0]
        coords0, feats0 = TG.voxelize_pc(input0, 64, time_index=0)
        coords1, feats1 = TG.voxelize_pc(input1, 64, time_index=1)
        coords = np.vstack([coords0, coords1])
        feats = np.vstack([feats0, feats1])

        coords = torch.from_numpy(coords).to(dtype=torch.int)
        feats = torch.from_numpy(feats).to(dtype=torch.float32)

        gt0 = self.targets_list[idx][1]
        gt1 = self.targets_list[idx][0]
        final_target, list_of_targets = DG.genarate_target(gt0, gt1, self.mode)
        
        return (coords, feats), (final_target, list_of_targets)

def mean_euclidean_distance(output, target):

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

def ReNet_train(model, mode, dataloaders, optimizer, scheduler, num_epochs, check_progress):

    print("Train start")

    model.train()
    
    bce_loss_fn = nn.BCELoss()
    euclidean_loss_fn = mean_euclidean_distance
    
    for epoch in range(num_epochs):

        total_loss1 = 0.0
        dataloader_index = 1

        for dataloader in dataloaders:
        
            total_loss2 = 0.0
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
                
                total_loss2 += loss.item()
                
                if check_progress:
                    print(f"Epoch [{epoch+1}/{num_epochs}], batch [{batch_index}/{len(dataloader)}] of datarloader [{dataloader_index}/{len(dataloaders)}]")
                batch_index += 1
            
            total_loss2 = total_loss2/len(dataloader)
            total_loss1 += total_loss2

            dataloader_index += 1
        
        print("-----------------------------------------")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss1/len(dataloaders)}")
        print("-----------------------------------------")

        scheduler.step()

def ReNet_evaluation(model, dataloaders):
    model.eval()
    
    euclidean_loss_fn = mean_euclidean_distance
    
    total_loss1 = 0.0
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for dataloader in dataloaders:
            
            total_loss2 = 0.0

            for batch in dataloader:
                
                input_data, targets = batch
                coords, feats = input_data
                final_target, _ = targets
                
                input_data = ME.SparseTensor(features=feats, coordinates=coords)
                
                final_output, _ = model(input_data)
                final_output = SP.sparse_to_dense_with_size(final_output, 64)
                
                med_loss = euclidean_loss_fn(final_output, final_target)
                
                total_loss2 += med_loss.item()

                final_pred = torch.argmax(final_output, dim=1)

                final_pred = final_pred.view(-1).cpu().numpy()
                final_target = final_target.view(-1).cpu().numpy()
                
                all_preds.extend(final_pred)
                all_targets.extend(final_target)

            total_loss2 = total_loss2 / len(dataloader)
            total_loss1 += total_loss2

    avg_loss = total_loss1 / len(dataloaders)
    
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')

    print(f"Evaluation Loss: {avg_loss}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    return avg_loss, precision, recall, f1

