import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from EnvioX.TerrainGenerator import TerrainGenerator # type: ignore
from EnvioX.SparseTensorProcessor import SparseTensorProcessor # type: ignore
from EnvioX.Visualizer import Visualizer # type: ignore

TG = TerrainGenerator()
SP = SparseTensorProcessor()

grid_size = 20
point_density_1 = 15
point_density_2 = 30 
detection_range = 3.2
robot_size = [0.4, 1.0, 0.8]
robot_speed = 1.0
voxel_resolution = 64

# generating environment config with designated number of obstacles
environment_config = {'grid_size': grid_size,  # length & width of environment (m)
                      'num_obstacles': {'num_boxes': 20, 'num_pillars': 10, 'num_walls': 5},
                      # number of obstacles which will be randomly spooned in environment
                      'point_density': point_density_1 # number of point per 1 meter
                      }

sensor_config = {'tilt_angle': 30, # downward tilt angle
                 'fov_angle': 80,
                 'detection_distance': 2,
                 'relative_position': {'front': [0.5, 0.0, 0.0], 
                                       'back': [-0.5, 0.0, 0.0], 
                                       'right': [0.0, -0.2, 0.0], 
                                       'left': [0.0, 0.2, 0.0]}
                 }

def main():
    # generating environment config with random number of obstacles
    environment_config_random = TG.generate_env_configs(grid_size=grid_size, 
                                                        point_density=point_density_1,
                                                        num_boxes_range=[20, 40],
                                                        num_pillars_range=[10, 20],
                                                        num_walls_range=[5, 10],
                                                        num_env_configs=1 # number of randomised environment configs
                                                                          # if it is bigger than 1,
                                                                          # generate_env_configs() generates array of environemnt config
                                                        )

    robot_positions, robot_yaws = TG.generate_robot_configs(grid_size=grid_size, 
                                                            detection_range=detection_range, # length & width & height of detection area (m)
                                                            robot_size=robot_size, # size of robot cosidering it as box [length, width, height] (m)
                                                            robot_speed=robot_speed, # speed of robot (m/s)
                                                            num_time_steps=0, # length of array of robot config
                                                            time_step=None # size of time step (s)
                                                            )

    # generating point cloud data of an environment
    environment = TG.generate_environment(env_config=environment_config,
                                          visualize=False
                                          )

    # generating point cloud data of a detection area
    terrain_data = TG.filter_points_in_detection_area(environment=environment,
                                                      detection_range=detection_range,
                                                      robot_size = robot_size,
                                                      robot_position=robot_positions[0],
                                                      visualize=True # if True, add 1 dummy point to the data
                                                                     # for well visualizing
                                                      )
    if terrain_data==None:
        print("Robot is at invalid position. Try again.")
        return False

    # generating sensor detected point cloud data of a detection area
    sensor_detection = TG.senser_detection(point_cloud=terrain_data,
                                           detection_range=detection_range,
                                           robot_size=robot_size,
                                           sensor_config=sensor_config,
                                           visualize=True
                                           )

    # generating voxelized data
    coords, feats = TG.voxelize_pc(point_cloud=sensor_detection, # return value of filter_points_in_detection_area() 
                                   voxel_resolution=voxel_resolution, # number of voxel in each side
                                   time_index=None # if 0 or 1,
                                                   # concatenates time index to coordinates
                                   )

    Visualizer.visualize_pc(terrain_data)
    Visualizer.visualize_pc(sensor_detection)
    Visualizer.visualize_voxel(coords, voxel_resolution)

if __name__ == "__main__":
    main()