import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import precision_score, recall_score, f1_score
from util.data_generation import DataGeneration
from util.data_processing import DataProcessing
from learning.model1 import ReNet1
from learning.model2 import ReNet2

DG = DataGeneration()
DP = DataProcessing()

class ReNetDataset(Dataset):

    def __init__(self, input_data_list, targets_list):
        self.input_data_list = input_data_list
        self.targets_list = targets_list
    
    def __len__(self):
        return len(self.input_data_list)
    
    def __getitem__(self, idx):

        DG = DataGeneration()

        input0 = self.input_data_list[idx][1]
        input1 = self.input_data_list[idx][0]
        coords0, feats0 = DG.voxelize_pc(input0, 64, time_index=0)
        coords1, feats1 = DG.voxelize_pc(input1, 64, time_index=1)
        coords = np.vstack([coords0, coords1])
        feats = np.vstack([feats0, feats1])

        coords = torch.from_numpy(coords).to(dtype=torch.int)
        feats = torch.from_numpy(feats).to(dtype=torch.float32)

        gt0 = self.targets_list[idx][1]
        gt1 = self.targets_list[idx][0]
        final_target, list_of_targets = DG.genarate_target(gt0, gt1)
        
        return (coords, feats), (final_target, list_of_targets)

def mean_euclidean_distance(output, target):

    diff = (output - target) ** 2
    sum_diff = torch.sum(diff, dim=1)
    distance = torch.sqrt(sum_diff)
    mean_distance = torch.mean(distance)
    
    return mean_distance

def collate_fn(batch):

    coords = []
    feats = []
    for item in batch:
        coords.append(item[0][0])
        feats.append(item[0][1])
    coords, feats = ME.utils.sparse_collate(coords, feats)
    final_targets = torch.stack([item[1][0] for item in batch])
    list_of_targets = [torch.stack([item[1][1][i] for item in batch]) for i in range(len(batch[0][1][1]))]
    
    return (coords, feats), (final_targets, list_of_targets)

def train(model, dataloaders, optimizer, scheduler, num_epochs):

    print("Train start")

    model.train()
    
    bce_loss_fn = nn.BCELoss()
    euclidean_loss_fn = mean_euclidean_distance
    
    for epoch in range(num_epochs):

        total_loss1 = 0.0

        for dataloader in dataloaders:
        
            total_loss2 = 0.0

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
                final_output = DP.sparse_to_dense_with_size(final_output, 64)
                final_output = final_output.squeeze()
                
                med_loss = euclidean_loss_fn(final_output, final_target)
                
                total_bce_loss = 0

                for output, target in zip(output_list,target_list):
                    output, _, _ = output.dense()
                    output = DP.dense_to_sparse(output)
                    b = output.C[:, 0]
                    x = output.C[:, 1]
                    y = output.C[:, 2]
                    z = output.C[:, 3]
                    t = output.C[:, 4]
                    target = target[b, x, y, z, t]
                    total_bce_loss += bce_loss_fn(output.F.squeeze(), target)
                                
                loss = med_loss + total_bce_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss2 += loss.item()
            
            total_loss2 = total_loss2/len(dataloader)
            total_loss1 += total_loss2
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss1/len(dataloaders)}")

        scheduler.step()

def evaluation(model, dataloaders):
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
                final_output = DP.sparse_to_dense_with_size(final_output, 64)
                
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

model = ReNet1(3, 3, 4, 0.5)
num_epochs = 2
batch_size = 2
optimizer = optim.Adam(model.parameters(), lr=0.01)
gamma = (0.0001 / 0.01) ** (1 / num_epochs)  
scheduler = ExponentialLR(optimizer, gamma=gamma)

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
                                                num_time_step=2,
                                                visualize=False
                                                )
print("Data is generated")

dataloaders = []

for i in range(1, len(input_lists) + 1):
    dataset = ReNetDataset(input_lists[f'list_{i}'], target_lists[f'list_{i}'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloaders.append(dataloader)

print("Dataloaders are set")

def train_test(dataloaders):
    
    for dataloader in dataloaders:

        for batch in dataloader:

            input_data, targets = batch
            coords, feats = input_data
            final_target, list_of_targets = targets

            input_data = ME.SparseTensor(features=feats, coordinates=coords)
            input_data, _, _ = input_data.dense()
                
            print("-----------------------")
            for i in range(len(list_of_targets)):
                target0 = list_of_targets[i][0, :, :, :, 0].squeeze()
                target1 = list_of_targets[i][0, :, :, :, 1].squeeze()
                #print(torch.sum(target))
                DG.visualize_voxel(target0, 2 ** (i + 3))
                DG.visualize_voxel(target1, 2 ** (i + 3))
            print("-----------------------")
                
#train_test(dataloaders)

train(model, dataloaders, optimizer, scheduler, num_epochs)