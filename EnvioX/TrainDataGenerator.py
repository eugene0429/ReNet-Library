import numpy as np
import torch
import MinkowskiEngine as ME
from .SparseTensorProcessor import SparseTensorProcessor as SP
from .TerrainGenerator import TerrainGenerator as TG
import json
import os

class TrainDataGenerator():
    
    @staticmethod
    def generate_dataset(grid_size,
                         detection_range,
                         robot_size,
                         robot_speed,
                         sensor_config,
                         point_density,
                         num_env_configs,
                         num_data_per_env,
                         num_time_step,
                         time_step,
                         save_path,
                         mode
                         ):

        env_configs = TG.generate_env_configs(grid_size,
                                              point_density,
                                              num_env_configs
                                              )
        
        targets_path = os.path.join(save_path, 'targets')
        inputs_path = os.path.join(save_path, 'inputs')
        os.makedirs(targets_path, exist_ok=True)
        os.makedirs(inputs_path, exist_ok=True)

        num_total_data = 0

        for env_config in env_configs:
            
            env = TG.generate_environment(env_config)
            num_data = 0

            while(num_data<num_data_per_env):
                
                robot_positions, _ = TG.generate_robot_configs(grid_size,
                                                               detection_range,
                                                               robot_size,
                                                               robot_speed,
                                                               num_time_step,
                                                               time_step
                                                               )
                for i in range(1, len(robot_positions)):
                    robot_position0 = robot_positions[i]
                    robot_position1 = robot_positions[i-1]

                    target0 = TG.filter_points_in_detection_area(env,
                                                                 detection_range,
                                                                 robot_size,
                                                                 robot_position0
                                                                 )
                                        
                    if target0 is None:
                        continue
                    
                    target1 = TG.filter_points_in_detection_area(env,
                                                                 detection_range,
                                                                 robot_size,
                                                                 robot_position1
                                                                 )
                    
                    if target1 is None:
                        continue

                    input0 = TG.senser_detection(target0,
                                                 detection_range,
                                                 robot_size,
                                                 sensor_config
                                                 )
                    
                    coords0_i, feats0_i = TG.voxelize_pc(input0, 64, time_index=0)
                    coords0_t, feats0_t = TG.voxelize_pc(target0, 64, time_index=0)
                    coords1_t, feats1_t = TG.voxelize_pc(target1, 64, time_index=1)

                    coords0_i = coords0_i.tolist()
                    feats0_i = feats0_i.tolist()
                    coords0_t = coords0_t.tolist()
                    feats0_t = feats0_t.tolist()
                    coords1_t = coords1_t.tolist()
                    feats1_t = feats1_t.tolist()

                    if mode == 'test':
                        target = [[coords0_t, feats0_t], [coords1_t, feats1_t]]

                    else:
                        targets = []
                        if mode == 1:
                            for i in range(4):
                                size = 2**(3+i)
                                coords0, _ = TG.voxelize_pc(target0, size, time_index=0)
                                coords1, _ = TG.voxelize_pc(target1, size, time_index=1)
                                coords = np.vstack([coords0, coords1])
                                coords = torch.tensor(coords)
                                target = torch.zeros((size, size, size, 2), dtype=torch.float32)
                                target[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = 1
                                target = target.tolist()
                                targets.append(target)
                        elif mode == 2:
                            for i in range(4):
                                size = 2**(3+i)
                                coords0, _ = TG.voxelize_pc(target0, size, time_index=None)
                                coords = torch.tensor(coords0)
                                target = torch.zeros((size, size, size), dtype=torch.float32)
                                target[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
                                target = target.tolist()
                                targets.append(target)
                        
                        target = [[coords0_t, feats0_t], [coords1_t, feats1_t], targets]

                    input = [coords0_i, feats0_i]

                    num_data += 1
                    num_total_data += 1
                    
                    target_path = os.path.join(targets_path, f'target_{num_total_data}.json')
                    input_path = os.path.join(inputs_path, f'input_{num_total_data}.json')

                    with open(target_path, 'w') as f:
                        json.dump(target, f)

                    with open(input_path, 'w') as f:
                        json.dump(input, f)

        return True