import numpy as np
import torch
import MinkowskiEngine as ME
from .TerrainGenerator import TerrainGenerator as TG

class SparseTensorProcessor():
    
    @staticmethod
    def print_sparse_tensor_num_coords_and_shape(sparse_tensor):
        num_coords = sparse_tensor.C.shape[0]
        dense_tensor, _, _ = sparse_tensor.dense()
        shape = dense_tensor.shape
        print("------------------------------------------------")
        print("number of points:", num_coords)
        print("shape of tensor:", shape)
        print("------------------------------------------------")
        return True

    @staticmethod
    def sparse_to_dense_with_size(sparse_tensor, size):
        coordinates = sparse_tensor.C
        features = sparse_tensor.F
        if coordinates.shape[1]==4:
            max_coordinates = torch.tensor([0] + [size - 1] * 3, dtype=torch.int)
        elif coordinates.shape[1]==5:
            max_coordinates = torch.tensor([0] + [size - 1] * 3 + [0], dtype=torch.int)
        
        if not (coordinates == max_coordinates).all(dim=1).any():
            zero_feature = torch.zeros((1, features.shape[1]), dtype=features.dtype)
            
            coordinates = torch.cat((coordinates, max_coordinates.unsqueeze(0)), dim=0)
            features = torch.cat((features, zero_feature), dim=0)
        
        updated_sparse_tensor = ME.SparseTensor(features, coordinates, tensor_stride=sparse_tensor.tensor_stride)

        dense_tensor, _, _ = updated_sparse_tensor.dense()
        
        return dense_tensor

    @staticmethod
    def dense_to_sparse(dense_tensor):
        if len(dense_tensor.shape) == 6:
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
        
        elif len(dense_tensor.shape) == 5:
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
    def concatenate_over_time_dimension(sparse_tensor):
        coords = sparse_tensor.C
        feats = sparse_tensor.F

        mask0 = coords[:, -1] == 0
        mask1 = coords[:, -1] == 1

        coords0 = coords[mask0]
        coords0 = coords0[:, :-1]
        feats0 = feats[mask0]

        coords1 = coords[mask1]
        coords1 = coords1[:, :-1]
        feats1 = feats[mask1]

        all_coords = torch.cat([coords0, coords1], dim=0).unique(dim=0)
        
        dict_A = {tuple(coord): i for i, coord in enumerate(coords0.tolist())}
        dict_B = {tuple(coord): i for i, coord in enumerate(coords1.tolist())}
        
        num_features_A = feats0.shape[1]
        num_features_B = feats1.shape[1]
        concatenated_features = []

        for coord in all_coords:
            coord_tuple = tuple(coord.tolist())
            features_A = feats0[dict_A[coord_tuple]] if coord_tuple in dict_A else torch.zeros(num_features_A)
            features_B = feats1[dict_B[coord_tuple]] if coord_tuple in dict_B else torch.zeros(num_features_B)
            concatenated_features.append(torch.cat([features_A, features_B], dim=0))
        
        concatenated_features = torch.stack(concatenated_features)

        tensor = ME.SparseTensor(
            features=concatenated_features,
            coordinates=all_coords,
            device=sparse_tensor.device
            # coordinate_manager=tensor_A.coordinate_manager
        )

        return tensor

    @staticmethod
    def generate_empty_sparse_tensor(sparse_tensor):
        coords = sparse_tensor.C
        feats = torch.zeros_like(sparse_tensor.F, dtype=torch.float32)
        out = ME.SparseTensor(
            features=feats, 
            coordinates=coords, 
            device=sparse_tensor.device,
            tensor_stride=sparse_tensor.tensor_stride,
            coordinate_manager=sparse_tensor.coordinate_manager
        )
        return out

    @staticmethod
    def reduce_dimension(sparse_tensor, stride):
        coords = sparse_tensor.C
        feats = sparse_tensor.F
        coords = coords[:, :-1].contiguous()
        out = ME.SparseTensor(
            features=feats, 
            coordinates=coords, 
            device=sparse_tensor.device,
            tensor_stride=stride
        )
        return out
    
    @staticmethod
    def pc_to_voxelized_sparse_tensor(point_cloud,
                                      voxel_resolution,
                                      time_index
                                      ):
        
        if len(point_cloud)==1:
            coords = []
            feats = []
            for i in range(len(point_cloud)):
                coord, feat = TG.voxelize_pc(point_cloud[i], voxel_resolution, time_index)
                coords.append(coord)
                feats.append(feat)
            coords, feats = ME.utils.sparse_collate(coords, feats)
            sparse_tensor = ME.SparseTensor(features=feats, coordinates=coords)

        else:
            coords, feats = TG.voxelize_pc(point_cloud, voxel_resolution, time_index)
            feats = torch.from_numpy(feats)
            coords = torch.from_numpy(coords)
            coords, feats = ME.utils.sparse_collate([coords], [feats])
            sparse_tensor = ME.SparseTensor(features=feats, coordinates=coords)

        return sparse_tensor

    
    def are_sparse_tensors_equal(self, tensor1, tensor2):
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

    def create_random_sparse_tensor(self, len, size, num_feature, dimension, stride=1):

        if dimension==3.5:
            coords = torch.randint(0, size, (len, 3))
            feats = torch.rand(len, num_feature, dtype=torch.float32)

            new_col = torch.cat([
                torch.zeros(len//2, 1, dtype=torch.int),
                torch.ones(len//2, 1, dtype=torch.int)
            ], dim=0)

            coords = torch.cat([coords, new_col], dim=1)

            coords, feats = ME.utils.sparse_collate([coords], [feats])

            s = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=stride)

        else:
            coords = torch.randint(0, size, (len, dimension))
            feats = torch.rand(len, num_feature)

            coords, feats = ME.utils.sparse_collate([coords], [feats])

            s = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=stride)

        return s