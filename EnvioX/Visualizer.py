import matplotlib.pyplot as plt
import torch
import numpy as np
import json
import os
import MinkowskiEngine as ME

class Visualizer():
    
    @staticmethod
    def visualize_pc(point_cloud):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], c=point_cloud[:,2], cmap='viridis', s=1)

        ax.grid(False)
        ax.axis('off')

        plt.show()

        return True
    
    @staticmethod
    def visualize_sparse_tensor(sparse_tensor,
                                voxel_resolution,
                                batch_index=0,
                                time_index=0
                                ):

        data = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution))
        coord = sparse_tensor.C
        if coord.shape[1] == 4:
            mask = (coord[:, 0] == batch_index)
        else:
            mask = (coord[:, 0] == batch_index) & (coord[:, 4] == time_index)
        coord = coord[mask]
        coord = coord.long()
        data[coord[:, 1], coord[:, 2], coord[:, 3]] = 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.voxels(data, edgecolor='k')

        ax.grid(False)
        ax.axis('off')

        plt.show()

        return True
    
    @staticmethod
    def visualize_voxel(coords,
                        voxel_resolution
                        ):

        data = np.zeros([voxel_resolution, voxel_resolution, voxel_resolution])
        coords = coords.long()
        data[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.voxels(data, edgecolor='k')

        ax.grid(False)
        ax.axis('off')

        plt.show()

        return True
    
    @staticmethod
    def visualize_json(file_path):
        
        with open(file_path, 'r') as f:
            data = json.load(f)

        if len(data) == 2:
            coords = data[0]
        else:
            coords = data[0][0]

        coords = np.array(coords)

        data = np.zeros([64, 64, 64])
        data[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.voxels(data, edgecolor='k')

        ax.grid(False)
        ax.axis('off')

        plt.show()

    @staticmethod
    def visualize_model_result(model,
                               model_path,
                               data_path,
                               data_index
                               ):

        model.load_state_dict(torch.load(model_path))
        model.eval()

        input_path = os.path.join(data_path, f'inputs/input_{data_index}.json')
        target_path = os.path.join(data_path, f'targets/target_{data_index}.json')

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
            model = model.to(device)
            output, _ = model(input)

        Visualizer.visualize_sparse_tensor(output, 64)
        Visualizer.visualize_voxel(coords0_t, 64)