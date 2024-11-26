import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import os
from EnvioX.TrainDataGenerator import generate_dataset # type: ignore

grid_size = 20
detection_range = 3.2
robot_size = [0.4, 1.0, 0.8]
robot_speed = 1.0
sensor_config = {'tilt_angle': 30,
                 'fov_angle': 80, 
                 'detection_distance': 2,
                 'relative_position': {'front': [0.5, 0.0, 0.0], 
                                       'back': [-0.5, 0.0, 0.0], 
                                       'right': [0.0, -0.2, 0.0], 
                                       'left': [0.0, 0.2, 0.0]}
                 }
point_density = 15
num_env_configs = 2
num_robot_per_env = 10
num_time_steps = 2
time_step_size = 0.2
model_mode = 1

def main():
    data_path = "data"
    os.makedirs(data_path, exist_ok=True)

    train_data_path = os.path.join(data_path, 'train')
    train_data_ratio = 0.7
    val_data_path = os.path.join(data_path, 'val')
    val_data_ratio = 0.2
    test_data_path = os.path.join(data_path, 'test')
    test_data_ratio = 0.1

    data_config = [('train', train_data_path, train_data_ratio),
                   ('val', val_data_path, val_data_ratio),
                   ('test', test_data_path, test_data_ratio)]

    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(val_data_path, exist_ok=True)
    os.makedirs(test_data_path, exist_ok=True)

    for type, path, ratio in data_config:

        generate_dataset(grid_size,
                         detection_range,
                         robot_size,
                         robot_speed,
                         sensor_config,
                         point_density,
                         num_env_configs,
                         round(num_robot_per_env * ratio),
                         num_time_steps,
                         time_step_size,
                         model_mode,
                         path,
                         type
                         )
    
    print("-----------------------------------------")
    print("           Data is generated             ")
    print("-----------------------------------------")

if __name__=="__main__":
    main()