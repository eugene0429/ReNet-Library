import numpy as np
import torch
import random
import math
import MinkowskiEngine as ME
from data_processing import DataProcessing
from data_generation import DataGeneration

DP = DataProcessing()
DG = DataGeneration()

grid_size = 20

sensors_config = {'tilt_angle': 30, 
                 'fov_angle': 80, 
                 'detection_distance': 2,
                 'relative_position': {'front': [0.5, 0.0, 0.0], 
                                       'back': [-0.5, 0.0, 0.0], 
                                       'right': [0.0, -0.2, 0.0], 
                                       'left': [0.0, 0.2, 0.0]}
                 }

robot_pose = DG.generate_robot_pose(grid_size=grid_size,
                                    detection_range=3.2,
                                    robot_height=0.8,
                                    robot_speed=1.0,
                                    synthetic=True)
robot_configs = []
for i in range(len(robot_pose)):
    robot_config = {'position': robot_pose[i]['position'],
                    'yaw': robot_pose[i]['yaw'],
                    'detection_range': 3.2,
                    'size': [0.4, 1.0, 0.8],
                    'sensors': sensors_config
                    }
    robot_configs.append(robot_config)

environment_config = {'grid_size': grid_size, 
                      'num_obstacles': {'num_boxes': 20, 'num_pillars': 1, 'num_walls': 1},
                      'point_density': 15
                      }

# test_pc = np.vstack([[x, y, z] for x in np.linspace(-1.6, 1.6, 30)
#                                for y in np.linspace(-1.6, 1.6, 30)
#                                for z in np.linspace(-1.6, 1.6, 30)])
# DG.visualize_pc(test_pc)

# test = DG.senser_detection(test_pc, test_robot_config, sensor_config)
# DG.visualize_pc(test)

environment = DG.generate_environment(environment_config)

data = DG.filter_points_in_detection_area(environment, robot_configs)

#sensor_detection = DG.senser_detection(data, robot_configs)

# for i in range(len(data)):
#     DG.visualize_pc(data[i])
#     DG.visualize_pc(sensor_detection[i])

# voxel_resolution = 64
# voxel_data = DG.pc_to_sparse_tensor(sensor_detection[0], voxel_resolution, time_index=None)
# DG.visualize_sparse_tensor_as_voxel(voxel_data)
