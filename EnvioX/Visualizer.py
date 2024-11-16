import matplotlib.pyplot as plt
import torch
import numpy as np
import json

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
    def visualize_sparse_tensor(data,
                                voxel_resolution,
                                batch_index=0,
                                time_index=0
                                ):

        _data = torch.zeros((voxel_resolution, voxel_resolution, voxel_resolution))
        coord = data.C
        if coord.shape[1] == 4:
            mask = (coord[:, 0] == batch_index)
        else:
            mask = (coord[:, 0] == batch_index) & (coord[:, 4] == time_index)
        coord = coord[mask]
        _data[coord[:, 1], coord[:, 2], coord[:, 3]] = 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.voxels(_data, edgecolor='k')

        ax.grid(False)
        ax.axis('off')

        plt.show()

        return True
    
    @staticmethod
    def visualize_voxel(coords,
                        voxel_resolution
                        ):

        data = np.zeros([voxel_resolution, voxel_resolution, voxel_resolution])
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
        