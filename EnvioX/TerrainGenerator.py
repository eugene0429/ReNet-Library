import numpy as np
import math
import random
from scipy.spatial.transform import Rotation as R

def _generate_box_point_cloud(center,
                              width,
                              length,
                              height,
                              point_density):

    x = np.linspace(center[0] - width / 2, center[0] + width / 2, int(point_density * width))
    y = np.linspace(center[1] - length / 2, center[1] + length / 2, int(point_density * length))
    z = np.linspace(0, height, int(point_density * height))

    X, Y, Z = np.meshgrid(x, y, z)

    points = np.vstack([
        np.column_stack((X.ravel(), Y.ravel(), np.full_like(Z.ravel(), 0))),
        np.column_stack((X.ravel(), Y.ravel(), np.full_like(Z.ravel(), height))),
        np.column_stack((X.ravel(), np.full_like(Y.ravel(), center[1] - length / 2), Z.ravel())),
        np.column_stack((X.ravel(), np.full_like(Y.ravel(), center[1] + length / 2), Z.ravel())),
        np.column_stack((np.full_like(X.ravel(), center[0] - width / 2), Y.ravel(), Z.ravel())),
        np.column_stack((np.full_like(X.ravel(), center[0] + width / 2), Y.ravel(), Z.ravel()))
    ])

    return points
    
def _generate_box_point_cloud_(center,
                               width,
                               length,
                               height,
                               point_density
                               ):

    x = np.linspace(center[0] - width / 2, center[0] + width / 2, int(point_density * width))
    y = np.linspace(center[1] - length / 2, center[1] + length / 2, int(point_density * length))
    z = np.linspace(0, height, math.ceil(int(point_density * width) * height / width))

    X, Y, Z = np.meshgrid(x, y, z)

    points = np.vstack([
        np.column_stack((X.ravel(), Y.ravel(), np.full_like(Z.ravel(), 0))),
        np.column_stack((X.ravel(), Y.ravel(), np.full_like(Z.ravel(), height))),
        np.column_stack((X.ravel(), np.full_like(Y.ravel(), center[1] - length / 2), Z.ravel())),
        np.column_stack((X.ravel(), np.full_like(Y.ravel(), center[1] + length / 2), Z.ravel())),
        np.column_stack((np.full_like(X.ravel(), center[0] - width / 2), Y.ravel(), Z.ravel())),
        np.column_stack((np.full_like(X.ravel(), center[0] + width / 2), Y.ravel(), Z.ravel()))
    ])

    return points

def _generate_multiple_boxes(num_boxes,
                             grid_size,
                             point_density
                             ):

    all_points = []

    for _ in range(num_boxes):
        width = np.random.uniform(0.2, 2.0)
        length = np.random.uniform(0.2, 2.0)
        height = np.random.uniform(0.08, 0.25)

        center = np.random.uniform(-(grid_size / 2- 1), grid_size / 2 - 1, 2)

        points = _generate_box_point_cloud_(center, width, length, height, point_density)

        all_points.append(points)

    return np.vstack(all_points)

def _generate_multiple_pillars(num_pillars,
                               grid_size,
                               point_density
                               ):

    all_points = []

    for _ in range(num_pillars):
        width = np.random.uniform(0.2, 1)
        length = np.random.uniform(0.2, 1)
        height = 4

        center = np.random.uniform(-(grid_size / 2 - 1), grid_size / 2 - 1, 2)

        points = _generate_box_point_cloud(center, width, length, height, point_density)

        all_points.append(points)

    return np.vstack(all_points)

def _generate_multiple_walls(num_walls,
                             grid_size,
                             point_density
                             ):

    all_points = []

    for _ in range(num_walls):

        orientation = random.choice(['horizontal', 'vertical'])

        if orientation == 'horizontal':
            width = np.random.uniform(6, 10)
            length = np.random.uniform(0.2, 0.5)
            height = 4

        else:
            width = np.random.uniform(0.2, 0.5)
            length = np.random.uniform(6, 10)
            height = 4

        center = np.random.uniform(-(grid_size / 2 - 5), grid_size / 2 - 5, 2)

        points = _generate_box_point_cloud(center, width, length, height, point_density)

        all_points.append(points)

    return np.vstack(all_points)

def _generate_ground(grid_size,
                     point_density
                     ):
        
    width = grid_size
    length = grid_size

    x = np.linspace(- width / 2, width / 2, int(point_density * width))
    y = np.linspace(- length / 2, length / 2, int(point_density * length))

    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X.ravel())

    points = np.column_stack((X.ravel(), Y.ravel(), Z))

    return points

def _rotate_vecter(vector, yaw, dimension):

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
        
    if dimension==2:
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
            ])
    elif dimension==3:
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
            ])
            
    rotated_vecter = vector @ rotation_matrix

    return rotated_vecter

def _check_vaildation(point_cloud, robot_position, robot_size):

    x_min, x_max = robot_position[0] - robot_size[0] / 1.5, robot_position[0] + robot_size[0] / 1.5
    y_min, y_max = robot_position[1] - robot_size[1] / 1.5, robot_position[1] + robot_size[1] / 1.5
    z_min, z_max = 0.5, 3.2

    if point_cloud.ndim==2:
        in_range = np.any(
            (point_cloud[:, 0] > x_min) & (point_cloud[:, 0] < x_max) &
            (point_cloud[:, 1] > y_min) & (point_cloud[:, 1] < y_max) &
            (point_cloud[:, 2] > z_min) & (point_cloud[:, 2] < z_max)
        )
    else:
        in_range = np.any(
            (point_cloud[0] > x_min) & (point_cloud[0] < x_max) &
            (point_cloud[1] > y_min) & (point_cloud[1] < y_max) &
            (point_cloud[2] > z_min) & (point_cloud[2] < z_max)
        )

    return in_range

def _is_in_fov(point, 
               sensor_position, 
               sensor_direction, 
               fov_angle, 
               max_distance
               ):

    vector = np.array(point) - np.array(sensor_position)
    distance = np.linalg.norm(vector)
        
    vector_norm = vector / distance
    sensor_direction_norm = sensor_direction / np.linalg.norm(sensor_direction)
        
    angle = np.arccos(np.clip(np.dot(vector_norm, sensor_direction_norm), -1.0, 1.0))
        
    return angle <= fov_angle / 2 and distance <= max_distance

def _generate_sensors(tilt_angle,
                      fov_angle,
                      detection_distance, positions
                      ):

    tilt_angle_rad = np.radians(tilt_angle)
    fov_angle_rad = np.radians(fov_angle)

    horizontal_component = np.cos(tilt_angle_rad)
    vertical_component = np.sin(tilt_angle_rad)

    sensors = [
            {'position': positions['front'],
             'direction': [horizontal_component, 0.0, -vertical_component],
             'fov': fov_angle_rad,
             'max_distance': detection_distance},
            {'position': positions['back'],
             'direction': [-horizontal_component, 0.0, -vertical_component],
             'fov': fov_angle_rad,
             'max_distance': detection_distance},
            {'position': positions['right'],
             'direction': [0.0, -horizontal_component, -vertical_component],
             'fov': fov_angle_rad,
             'max_distance': detection_distance},
            {'position': positions['left'],
             'direction': [0.0, horizontal_component, -vertical_component],
             'fov': fov_angle_rad,
             'max_distance': detection_distance},
        ]

    return sensors

def _noisify_point_cloud(point_cloud,
                         robot_position,
                         robot_size, 
                         detection_range
                         ):

    POS_NOISE_RANGE = 0.05
    TILT_ANGLE_RANGE = 1
    HEIGHT_NOISE_RANGE = 0.05
    PRUNING_PERCENTAGE = random.uniform(0.05, 0.1)
    NUM_CLUSTERS = random.randint(3, 6)
    POINTS_PER_CLUSER = random.randint(10, 20)

    position_noise = np.random.uniform(-POS_NOISE_RANGE, POS_NOISE_RANGE, point_cloud.shape)
    point_cloud += position_noise

    tilt_angle = np.radians(np.random.uniform(-TILT_ANGLE_RANGE, TILT_ANGLE_RANGE))
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    rotation = R.from_rotvec(axis * tilt_angle)
    point_cloud = rotation.apply(point_cloud)

    patch_indices = np.random.choice(point_cloud.shape[0], size=int(0.2 * point_cloud.shape[0]), replace=False)
    height_noise = np.random.uniform(-HEIGHT_NOISE_RANGE, HEIGHT_NOISE_RANGE, len(patch_indices))
    point_cloud[patch_indices, 2] += height_noise

    keep_indices = np.random.choice(point_cloud.shape[0], size=int((1-PRUNING_PERCENTAGE) * point_cloud.shape[0]), replace=False)
    point_cloud = point_cloud[keep_indices, :]

    cluster_center_range = [[robot_position[0] - detection_range / 2.5, robot_position[0] + detection_range / 2.5],
                                [robot_position[1] - detection_range / 2.5, robot_position[1] + detection_range / 2.5],
                                [0, detection_range / 2]]

    for _ in range(NUM_CLUSTERS):
        cluster_center = np.array([random.uniform(cluster_center_range[0][0], cluster_center_range[0][1]),
                                       random.uniform(cluster_center_range[1][0], cluster_center_range[1][1]),
                                       random.uniform(cluster_center_range[2][0], cluster_center_range[2][1])])
        if _check_vaildation(cluster_center,
                             robot_position,
                             robot_size
                             ):
            continue

        cluster_points = cluster_center + np.random.normal(scale=0.08, size=(POINTS_PER_CLUSER, 3))
        point_cloud = np.vstack((point_cloud, cluster_points))
        
    return point_cloud

class TerrainGenerator():

    @staticmethod
    def generate_environment(env_config):

        grid_size = env_config['grid_size']
        num_obstacles = env_config['num_obstacles']
        point_density = env_config['point_density']

        ground = _generate_ground(grid_size, point_density)
        boxes = _generate_multiple_boxes(num_obstacles['num_boxes'], grid_size, point_density)
        pillars = _generate_multiple_pillars(num_obstacles['num_pillars'], grid_size, point_density)
        walls = _generate_multiple_walls(num_obstacles['num_walls'], grid_size, point_density)

        environment = np.vstack([ground, boxes, pillars, walls])

        return environment
    
    @staticmethod
    def generate_env_configs(grid_size,
                             point_density,
                             num_env_configs
                             ):

        env_configs = []

        for i in range(num_env_configs):
            num_boxes = random.randint(20, 40)
            num_pillars = random.randint(8, 16)
            num_walls = random.randint(4, 8)
            env_config = {'grid_size': grid_size, 
                          'num_obstacles': {'num_boxes': num_boxes, 
                                            'num_pillars': num_pillars, 
                                            'num_walls': num_walls},
                          'point_density': point_density
                          }
            env_configs.append(env_config)
        
        if num_env_configs==1:
            return env_configs[0]
        else:
            return env_configs
    
    @staticmethod
    def generate_robot_configs(grid_size, 
                               detection_range, 
                               robot_size, 
                               robot_speed, 
                               sensors_config,
                               time_step,
                               num_time_step
                               ):
        
        range_value = grid_size / 2 - detection_range

        if time_step != None:
            robot_positions = []
            robot_yaws = []
            for i in range(num_time_step + 1):
                if i==0:
                    robot_position = np.array([random.uniform(-range_value, range_value), 
                                               random.uniform(-range_value, range_value), 
                                               robot_size[2]]
                                               )
                    robot_yaw = random.uniform(-math.pi, math.pi)

                    robot_positions.append(robot_position)
                    robot_yaws.append(robot_yaw)
                    
                else:
                    direction_vector = np.array([np.cos(robot_yaw), np.sin(robot_yaw), 0])
                    robot_position = robot_position + robot_speed * direction_vector * time_step
                    robot_yaw = robot_yaw + random.uniform(-math.pi / 10, math.pi / 10)
                    
                    robot_positions.append(robot_position)
                    robot_yaws.append(robot_yaw)
            
            robot_config = {'position': robot_positions,
                            'yaw': robot_yaws,
                            'detection_range': detection_range,
                            'size': robot_size,
                            'sensors': sensors_config
                            }

            return robot_config
            
        else:
            robot_position = np.array([random.uniform(-range_value, range_value), 
                                       random.uniform(-range_value, range_value), 
                                       robot_size[2]]
                                       )
            robot_yaw = random.uniform(- math.pi, math.pi)
            
            robot_config = {'position': [robot_position],
                            'yaw': [robot_yaw],
                            'detection_range': detection_range,
                            'size': robot_size,
                            'sensors': sensors_config
                            }

            return robot_config

    @staticmethod
    def filter_points_in_detection_area(environment, 
                                        robot_config, 
                                        use_yaw=True, 
                                        visualize=False
                                        ):
        
        detection_range = robot_config['detection_range']
        robot_size = robot_config['size']

        points = []

        for i in range(len(robot_config['position'])):

            robot_position = robot_config['position'][i]
            robot_yaw = robot_config['yaw'][i]

            if not use_yaw:
                robot_yaw = 0
            
            if _check_vaildation(environment,
                                 robot_position,
                                 robot_size
                                 ):
                return None

            translated_point_cloud = environment[:, :2] - robot_position[:2]
            
            rotated_point_cloud = _rotate_vecter(vector=translated_point_cloud,
                                                 yaw=robot_yaw,
                                                 dimension=2
                                                 )

            rotated_point_cloud = np.hstack((rotated_point_cloud, environment[:, 2:3]))
            
            x_min = -detection_range / 2
            x_max = detection_range / 2
            y_min = -detection_range / 2
            y_max = detection_range / 2
            z_min = 0
            z_max = 3.2

            filtered_points = rotated_point_cloud[
                (rotated_point_cloud[:, 0] >= x_min) & (rotated_point_cloud[:, 0] <= x_max) &
                (rotated_point_cloud[:, 1] >= y_min) & (rotated_point_cloud[:, 1] <= y_max) &
                (rotated_point_cloud[:, 2] >= z_min) & (rotated_point_cloud[:, 2] <= z_max)
            ]

            filtered_points[:, :2] = _rotate_vecter(vector=filtered_points[:, :2],
                                                    yaw=robot_yaw,
                                                    dimension=2
                                                    ) + robot_position[:2]
            
            if visualize:
                filtered_points = np.vstack([filtered_points,
                                             np.array([robot_position[0],
                                                       robot_position[1],
                                                       detection_range])])
                                                       
            points.append(filtered_points)

        return points

    @staticmethod
    def senser_detection(point_clouds,
                         robot_config,
                         visualize=False
                         ):
        
        detection_range = robot_config['detection_range']
        robot_size = robot_config['size']
        sensor_config = robot_config['sensors']
        sensors = _generate_sensors(sensor_config['tilt_angle'], 
                                    sensor_config['fov_angle'], 
                                    sensor_config['detection_distance'], 
                                    sensor_config['relative_position']
                                    )

        points = []

        for i in range(len(point_clouds)):

            robot_position = robot_config['position'][i]
            robot_position = robot_position + np.array([random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), 0])
            robot_yaw = robot_config['yaw'][i]

            detected_points = []
            
            for point in point_clouds[i]:
                visible = False
                for sensor in sensors:
                    rotated_position = _rotate_vecter(vector=np.array(sensor['position']),
                                                      yaw=robot_yaw,
                                                      dimension=3
                                                      )
                    sensor_position = robot_position + rotated_position
                    rotated_direction = _rotate_vecter(vector=np.array(sensor['direction']),
                                                       yaw=robot_yaw,
                                                       dimension=3
                                                       )
                    if _is_in_fov(point, sensor_position,
                                  rotated_direction, sensor['fov'],
                                  sensor['max_distance']
                                  ):
                        visible = True
                        break
                if visible:
                    detected_points.append(point)

            detected_points = np.array(detected_points)
            
            detected_points = _noisify_point_cloud(detected_points,
                                                   robot_position,
                                                   robot_size,
                                                   detection_range
                                                   )

            if visualize:
                detected_points = np.vstack([detected_points,
                                             np.array([robot_position[0], robot_position[1], detection_range])])

            points.append(detected_points)
        
        return points
    
    @staticmethod
    def voxelize_pc(point_cloud,
                    voxel_resolution,
                    time_index
                    ):

        
        points = point_cloud
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        points_normalized = (points - min_coords) / (max_coords - min_coords + 1e-15)
        points_scaled = voxel_resolution * points_normalized

        voxel_indices = np.floor(points_scaled).astype(np.int32)

        voxel_keys, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

        coords = voxel_keys
        
        if not time_index==None:
            if time_index == 0:
                coords = np.hstack((voxel_keys, np.zeros((len(voxel_keys), 1), dtype=np.int32)))
            elif time_index == 1:
                coords = np.hstack((voxel_keys, np.ones((len(voxel_keys), 1), dtype=np.int32)))
            else:
                print("time index should be 0 or 1")
                return None

        centroids = np.array([points_scaled[inverse_indices == i].mean(axis=0) for i in range(len(voxel_keys))])
        centroids = centroids % 1
        feats = centroids

        return coords, feats