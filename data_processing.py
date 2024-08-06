import numpy as np
import torch
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

class data_processing():
    
    @staticmethod
    def sparse_to_dense_with_size(sparse_tensor, size):
        coordinates = sparse_tensor.C
        features = sparse_tensor.F
        
        max_coordinates = torch.tensor([0] + [size - 2] * (coordinates.shape[1] - 1) + [0], dtype=torch.int)
        
        if not (coordinates == max_coordinates).all(dim=1).any():
            zero_feature = torch.zeros((1, features.shape[1]), dtype=features.dtype)
            
            coordinates = torch.cat((coordinates, max_coordinates.unsqueeze(0)), dim=0)
            features = torch.cat((features, zero_feature), dim=0)
        
        updated_sparse_tensor = ME.SparseTensor(features, coordinates, tensor_stride=sparse_tensor.tensor_stride)

        dense_tensor, _, _ = updated_sparse_tensor.dense()
        
        return dense_tensor

    def dense_to_sparse(self, dense_tensor, input_dimension):
        if input_dimension == 4:
            batch_size, channel_dimension, width, height, depth, time = dense_tensor.shape

            b, x, y, z, t = torch.meshgrid(
                torch.arange(batch_size),
                torch.arange(width),
                torch.arange(height),
                torch.arange(depth),
                torch.arange(time),
                indexing='ij'
            )

            coords = torch.stack([b, x, y, z, t], dim=-1)

            dense_tensor = dense_tensor.permute(0, 2, 3, 4, 5, 1).reshape(-1, channel_dimension)
            coords = coords.reshape(-1, 5)
        
        elif input_dimension == 3:
            batch_size, channel_dimension, width, height, depth = dense_tensor.shape

            b, x, y, z = torch.meshgrid(
                torch.arange(batch_size),
                torch.arange(width),
                torch.arange(height),
                torch.arange(depth),
                indexing='ij'
            )

            coords = torch.stack([b, x, y, z], dim=-1)

            dense_tensor = dense_tensor.permute(0, 2, 3, 4, 1).reshape(-1, channel_dimension)
            coords = coords.reshape(-1, 4)
        
        else:
            raise ValueError("input_dimension must be either 3 or 4")

        non_zero_mask = torch.any(dense_tensor != 0, dim=1)

        non_zero_coords = coords[non_zero_mask]
        non_zero_features = dense_tensor[non_zero_mask]

        sparse_tensor = ME.SparseTensor(features=non_zero_features.float(), coordinates=non_zero_coords.int())
        return sparse_tensor

    @staticmethod
    def concatenate_sparse_tensors(tensor_A, tensor_B, stride):
        coords_A = tensor_A.C
        coords_B = tensor_B.C
        all_coords = torch.cat([coords_A, coords_B], dim=0).unique(dim=0)
        
        dict_A = {tuple(coord): i for i, coord in enumerate(coords_A.tolist())}
        dict_B = {tuple(coord): i for i, coord in enumerate(coords_B.tolist())}
        
        num_features_A = tensor_A.F.shape[1]
        num_features_B = tensor_B.F.shape[1]
        concatenated_features = []

        for coord in all_coords:
            coord_tuple = tuple(coord.tolist())
            features_A = tensor_A.F[dict_A[coord_tuple]] if coord_tuple in dict_A else torch.zeros(num_features_A)
            features_B = tensor_B.F[dict_B[coord_tuple]] if coord_tuple in dict_B else torch.zeros(num_features_B)
            concatenated_features.append(torch.cat([features_A, features_B], dim=0))
        
        concatenated_features = torch.stack(concatenated_features)
        out = ME.SparseTensor(
            features=concatenated_features, 
            coordinates=all_coords, 
            device=tensor_A.device,
            tensor_stride=stride,
            coordinate_manager=tensor_A.coordinate_manager
        )

        return out
    
    @staticmethod
    def noisify_point_cloud(coo):

        point_cloud = np.array(coo).T

        POS_NOISE_RANGE = 0.05
        TILT_ANGLE_RANGE = 1
        HEIGHT_NOISE_RANGE = 0.05
        PRUNING_PERCENTAGE = 0.1
        OUTLIERS_COUNT = 100
        ROBOT_POSE_RANGE = 0.05

        # Position Noise
        position_noise = np.random.uniform(-POS_NOISE_RANGE, POS_NOISE_RANGE, point_cloud.shape)
        point_cloud += position_noise

        # Tilt Transformation
        tilt_angle = np.radians(np.random.uniform(-TILT_ANGLE_RANGE, TILT_ANGLE_RANGE))
        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = R.from_rotvec(axis * tilt_angle)
        point_cloud = rotation.apply(point_cloud.T).T 

        # Height Noise
        patch_indices = np.random.choice(point_cloud.shape[1], size=int(0.2 * point_cloud.shape[1]), replace=False)
        height_noise = np.random.uniform(-HEIGHT_NOISE_RANGE, HEIGHT_NOISE_RANGE, len(patch_indices))
        point_cloud[2, patch_indices] += height_noise

        # Pruning
        keep_indices = np.random.choice(point_cloud.shape[1], size=int((1-PRUNING_PERCENTAGE) * point_cloud.shape[1]), replace=False)
        point_cloud = point_cloud[:, keep_indices]

        # Outliers
        means = np.mean(point_cloud, axis=1).reshape(3, 1)
        stds = np.std(point_cloud, axis=1).reshape(3, 1)
        outliers = np.random.normal(means, stds, (3, OUTLIERS_COUNT))
        point_cloud = np.concatenate((point_cloud, outliers), axis=1)

        # Robot Pose Noise
        robot_pose = np.random.uniform(-ROBOT_POSE_RANGE, ROBOT_POSE_RANGE, 3)
        
        return point_cloud.T

    def voxelize_pc_with_time(self, coo, voxel_resolution, time_index):

        points = coo
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        points_normalized = (points - min_coords) / (max_coords - min_coords + 1e-15)
        points_scaled = voxel_resolution * points_normalized

        voxel_indices = np.floor(points_scaled).astype(np.int32)

        voxel_keys, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

        if time_index == 0:
            coord = np.hstack((voxel_keys, np.zeros((len(voxel_keys), 1), dtype=np.int32)))
        else:
            coord = np.hstack((voxel_keys, np.ones((len(voxel_keys), 1), dtype=np.int32)))

        centroids = np.array([points_scaled[inverse_indices == i].mean(axis=0) for i in range(len(voxel_keys))])
        centroids = centroids % 1
        feat = centroids

        return coord, feat
    
    def voxelize_pc(self, coo, voxel_resolution):

        points = coo
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        points_normalized = (points - min_coords) / (max_coords - min_coords + 1e-15)
        points_scaled = voxel_resolution * points_normalized

        voxel_indices = np.floor(points_scaled).astype(np.int32)

        voxel_keys, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
        coord = voxel_keys

        centroids = np.array([points_scaled[inverse_indices == i].mean(axis=0) for i in range(len(voxel_keys))])
        centroids = centroids % 1
        feat = centroids

        return coord, feat
        
    def coo_to_sparse_tensor(self, input, voxel_resolution):
        coord, feat = self.voxelize_pc(input, voxel_resolution)
        feat = torch.from_numpy(feat)
        coord = torch.from_numpy(coord)
        coord, feat = ME.utils.sparse_collate([coord], [feat])
        sparse_tensor = ME.SparseTensor(features=feat, coordinates=coord)
        return sparse_tensor

    def coo_to_sparse_tensor_with_time_batched(self, input, voxel_resolution, time_index):
        coords = []
        feats = []
        for i in range(len(input)):
            coord, feat = self.voxelize_pc_with_time(input[i], voxel_resolution, time_index)
            coords.append(coord)
            feats.append(feat)
    
        coords_cat, feats_cat = ME.utils.sparse_collate(coords, feats)
        sparse_tensor = ME.SparseTensor(features=feats_cat, coordinates=coords_cat)

        return sparse_tensor
    
    def coo_to_sparse_tensor_batched(self, input, voxel_resolution):
        coords = []
        feats = []
        for i in range(len(input)):
            coord, feat = self.voxelize_pc(input[i], voxel_resolution)
            coords.append(coord)
            feats.append(feat)
    
        coords_cat, feats_cat = ME.utils.sparse_collate(coords, feats)
        sparse_tensor = ME.SparseTensor(features=feats_cat, coordinates=coords_cat)

        return sparse_tensor
    
    def genarate_target(self, ground_truth):
        targets = []
        output_target = self.coo_to_sparse_tensor(ground_truth, 64)

        for i in range(4):
            coords, _ = self.voxelize_pc(ground_truth, 2**(3+i))
            feats = np.ones((coords.shape[0], 1), dtype=np.int8)
            coords, feats = ME.utils.sparse_collate([coords], [feats])
            target = ME.SparseTensor(features=feats, coordinates=coords)
            targets.append(target)
        
        return targets, output_target

    def visualize_pc(self, coo):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(coo[:,0], coo[:,1], coo[:,2], c=coo[:,2], cmap='viridis', s=1)

        ax.grid(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.axis('off')

        plt.show()

        return True
    
    def visualize_voxel(self, sparse_tensor):

        data = np.zeros((64, 64, 64))
        coord = sparse_tensor.C
        coord = coord.numpy()

        x_coords = coord[:, 1]
        y_coords = coord[:, 2]
        z_coords = coord[:, 3]

        data[x_coords, y_coords, z_coords] = 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.voxels(data, edgecolor='k')

        ax.grid(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.axis('off')

        plt.show()

        return True
    
    @staticmethod
    def are_tensors_equal(tensor1, tensor2):
        coords1, feats1 = tensor1.coordinates, tensor1.features
        coords2, feats2 = tensor2.coordinates, tensor2.features

        if coords1.size == 0 or coords2.size == 0:
            return coords1.size == coords2.size

        sorted_indices1 = np.lexsort(tuple(coords1[:, i] for i in range(coords1.shape[1]-1, -1, -1)))
        sorted_indices2 = np.lexsort(tuple(coords2[:, i] for i in range(coords2.shape[1]-1, -1, -1)))

        sorted_coords1 = coords1[sorted_indices1]
        sorted_feats1 = feats1[sorted_indices1]

        sorted_coords2 = coords2[sorted_indices2]
        sorted_feats2 = feats2[sorted_indices2]

        return np.array_equal(sorted_feats1, sorted_feats2), np.array_equal(sorted_coords1, sorted_coords2)

def generate_stair_shape(num_steps, width, depth, height, points_per_step):
    x_coords = []
    y_coords = []
    z_coords = []
    
    for i in range(num_steps):
        z = (i + 1) * height
        for w in np.linspace(0, width, points_per_step):
            for d in np.linspace(i * depth, (i + 1) * depth, points_per_step):
                x_coords.append(w)
                y_coords.append(d)
                z_coords.append(z)

        y = (i + 1) * depth
        for w in np.linspace(0, width, points_per_step):
            for h in np.linspace(i * height, (i + 1) * height, points_per_step):
                x_coords.append(w)
                y_coords.append(y)
                z_coords.append(h)

    return np.array([np.array(x_coords), np.array(y_coords), np.array(z_coords)])

    
DP = data_processing()

num_steps = 8
width = 3.2
depth = 0.4
height = 0.4
points_per_step = 20

# points = generate_stair_shape(num_steps, width, depth, height, points_per_step).T
# nosifed_points = DP.noisify_point_cloud(points)
# DP.visualize_pc(points)
# DP.visualize_pc(nosifed_points)

# sparse_tensor = DP.coo_to_sparse_tensor(nosifed_points, 64)
# DP.visualize_voxel(sparse_tensor)

# target = DP.genarate_target(points)
