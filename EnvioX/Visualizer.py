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
    def visualize_voxel(data,
                        voxel_resolution
                        ):

        _data = np.zeros([voxel_resolution, voxel_resolution, voxel_resolution])
        _data[data[:, 0], data[:, 1], data[:, 2]] = 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.voxels(_data, edgecolor='k')

        ax.grid(False)
        ax.axis('off')

        plt.show()

        return True
    
    @staticmethod
    def visualize_json(file_path,
                       time_index
                       ):
        
        with open(file_path, 'r') as f:
            point_cloud = json.load(f)

        point_cloud = np.array(point_cloud[time_index])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], c=point_cloud[:,2], cmap='viridis', s=1)

        ax.grid(False)
        ax.axis('off')

        plt.show()

        