import numpy as np
import torch
from .TerrainGenerator import TerrainGenerator as TG
import json
import os
    
def generate_dataset(grid_size,
                     detection_range,
                     robot_size,
                     robot_speed,
                     sensor_config,
                     point_density,
                     num_env_configs,
                     num_robot_per_env,
                     num_time_steps,
                     time_step_size,
                     model_mode,
                     save_path,
                     data_type
                     ):

    env_configs = TG.generate_env_configs(grid_size,
                                          point_density,
                                          num_env_configs
                                          )
    
    targets_path = os.path.join(save_path, 'targets')
    inputs_path = os.path.join(save_path, 'inputs')
    os.makedirs(targets_path, exist_ok=True)
    os.makedirs(inputs_path, exist_ok=True)

    data_idx = 0

    for env_config in env_configs:
        
        env = TG.generate_environment(env_config)
        num_robot = 0

        while(num_robot<num_robot_per_env):
            
            robot_positions, _ = TG.generate_robot_configs(grid_size,
                                                           detection_range,
                                                           robot_size,
                                                           robot_speed,
                                                           num_time_steps,
                                                           time_step_size
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

                if data_type == 'test' or data_type == 'val':
                    target = [[coords0_t, feats0_t], [coords1_t, feats1_t]]

                elif data_type == 'train':
                    targets = []
                    if model_mode == 1:
                        for i in range(4):
                            size = 2**(3+i)
                            coords0, _ = TG.voxelize_pc(target0, size, time_index=0)
                            coords1, _ = TG.voxelize_pc(target1, size, time_index=1)
                            coords = np.vstack([coords0, coords1])
                            target = np.zeros((size, size, size, 2))
                            target[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = 1
                            target = target.tolist()
                            targets.append(target)
                    elif model_mode == 2:
                        for i in range(4):
                            size = 2**(3+i)
                            coords0, _ = TG.voxelize_pc(target0, size, time_index=None)
                            target = np.zeros((size, size, size))
                            target[coords0[:, 0], coords0[:, 1], coords0[:, 2]] = 1
                            target = target.tolist()
                            targets.append(target)
                    
                    target = [[coords0_t, feats0_t], [coords1_t, feats1_t], targets]
                else:
                    print("data_type should be 'train' or 'val' or 'test'")
                    return False

                input = [coords0_i, feats0_i]

                data_idx += 1
                
                target_path = os.path.join(targets_path, f'target_{data_idx}.json')
                input_path = os.path.join(inputs_path, f'input_{data_idx}.json')

                with open(target_path, 'w') as f:
                    json.dump(target, f)

                with open(input_path, 'w') as f:
                    json.dump(input, f)
            
            num_robot += 1

    return True