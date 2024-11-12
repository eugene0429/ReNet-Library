import numpy as np
import torch
import MinkowskiEngine as ME
from .SparseTensorProcessor import SparseTensorProcessor as SP
from .TerrainGenerator import TerrainGenerator as TG

class TrainDataGenerator():
    
    @staticmethod
    def genarate_target(ground_truth_0,
                        ground_truth_1,
                        mode
                        ):

        targets = []
        output_target = SP.pc_to_voxelized_sparse_tensor(ground_truth_0, 64, time_index=0)
        output_target = SP.sparse_to_dense_with_size(output_target, 64)
        output_target = output_target.squeeze()

        if mode == 1:
            for i in range(4):
                size = 2**(3+i)
                coords0, _ = TG.voxelize_pc(ground_truth_0, size, time_index=0)
                coords1, _ = TG.voxelize_pc(ground_truth_1, size, time_index=1)
                coords = np.vstack([coords0, coords1])
                coords = torch.tensor(coords)
                target = torch.zeros((size, size, size, 2), dtype=torch.float32)
                target[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = 1
                targets.append(target)
        elif mode == 2:
            for i in range(4):
                size = 2**(3+i)
                coords0, _ = TG.voxelize_pc(ground_truth_0, size, time_index=None)
                coords = torch.tensor(coords0)
                target = torch.zeros((size, size, size), dtype=torch.float32)
                target[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
                targets.append(target)
        else:
            print("mode should be 1 or 2")
            return None
        
        return output_target, targets
    
    @staticmethod
    def generate_dataset(grid_size,
                         detection_range,
                         robot_size,
                         robot_speed,
                         sensors_config,
                         point_density,
                         num_env_configs,
                         num_data_per_env,
                         time_step,
                         num_time_step,
                         visualize=False
                         ):

        env_configs = TG.generate_env_configs(grid_size,
                                              point_density,
                                              num_env_configs
                                              )
        
        input_data_lists = {}
        target_lists = {}
        for i in range(1, num_time_step + 1):
            input_data_lists[f'list_{i}'] = []
            target_lists[f'list_{i}'] = []

        for env_config in env_configs:
            
            env = TG.generate_environment(env_config)
            num_data = 0

            while(num_data<num_data_per_env):
                
                robot_config = TG.generate_robot_configs(grid_size,
                                                         detection_range,
                                                         robot_size,
                                                         robot_speed,
                                                         sensors_config,
                                                         time_step,
                                                         num_time_step
                                                         )
                
                target = TG.filter_points_in_detection_area(env,
                                                            robot_config,
                                                            visualize
                                                            )

                if target==None:
                    continue
                
                input = TG.senser_detection(target,
                                            robot_config,
                                            visualize
                                            )

                for i in range(1, num_time_step + 1):
                    input_data_lists[f'list_{i}'].append(input[i-1:i+1])
                    target_lists[f'list_{i}'].append(target[i-1:i+1])

                num_data += 1

        return input_data_lists, target_lists