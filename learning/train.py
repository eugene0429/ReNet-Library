import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import torch
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from util.data_processing import DataProcessing as DP
from learning.model1 import Net1
from learning.model2 import Net2

class VoxelDataset(Dataset):
    def __init__(self, data_now, data_previous):
        self.data0 = data_now
        self.data1 = data_previous

    def __len__(self):
        return len(self.data0)
    
    def __getitem__(self, idx):
        pc0, targets = self.data0[idx]
        pc1 = self.data1[idx]
        pc0 = DP.noisify_point_cloud(pc0)
        coords0, feats0 = DP.voxelize_pc_with_time(pc0, 64, 0)
        coords1, feats1 = DP.voxelize_pc_with_time(pc1, 64, 1)
        coords = np.vstack(coords0, coords1)
        feats = np.vstack(feats0, feats1)
        return coords, feats, targets

data0 = []
data1 = []
dataset = VoxelDataset(data0, data1)
batch_size = 2
data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=ME.utils.batch_sparse_collate, shuffle=True)

model = Net1(3, 3, 4, 0.5)
num_epochs = 100
crit1 = torch.nn.MSELoss()
crit2 = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
gamma = (0.0001 / 0.01) ** (1 / num_epochs)  
scheduler = ExponentialLR(optimizer, gamma=gamma)

accum_loss, accum_iter, tot_iter = 0, 0, 0

for epoch in range(num_epochs):
    train_iter = iter(data_loader)
    model.train()
    for i, data in enumerate(train_iter):
        coords, feats, targets = data
        t2, t1 = targets

        input = ME.SparseTensor(feattures=feats, coordinates=coords)
        lhs, output = model(input)

        optimizer.zero_grad()

        output = DP.sparse_to_dense_with_size(output, 64).squeeze()
        loss1 = crit1(output, t1)
        l = 0
        for i in range(4):
            lh = DP.sparse_to_dense_with_size(lhs[i], 2 ** (3 + i))
            l += crit2(lh, t2[i])
        loss2 = l

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        accum_loss += loss.item()
        accum_iter += 1
        tot_iter += 1

        if tot_iter % 10 == 0 or tot_iter == 1:
            print(
                f'Epoch: {epoch} iter: {tot_iter}, Loss: {accum_loss / accum_iter}'
            )
            accum_loss, accum_iter = 0, 0   

    scheduler.step()