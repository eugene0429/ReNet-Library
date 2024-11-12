import matplotlib.pyplot as plt
import torch
import numpy as np

class Visualizer():
    
    @staticmethod
    def visualize_pc(point_clouds):
        
        for i in range(len(point_clouds)):
            point_cloud = point_clouds[i]
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