import numpy as np
import math
import random
from scipy.spatial.transform import Rotation as R
    
def _generate_box_point_cloud(center,
                              width,
                              length,
                              height,
                              point_density,
                              mode
                              ):

    if mode == 1:
        num_x = int(point_density * width)
        num_z = math.ceil(int(point_density * width) * height / width)
    elif mode == 2:
        num_x = int(point_density * width)
        num_z = int(point_density * height)
    elif mode == 3:
        num_x = math.ceil(int(point_density * height) * width / height)
        num_z = int(point_density * height)

    x = np.linspace(- width / 2, width / 2, num_x)
    y = np.linspace(- length / 2, length / 2, int(point_density * length))
    z = np.linspace(0, height, num_z)

    X, Y, Z = np.meshgrid(x, y, z)

    points = np.vstack([
        np.column_stack((X.ravel(), Y.ravel(), np.full_like(Z.ravel(), 0))),
        np.column_stack((X.ravel(), Y.ravel(), np.full_like(Z.ravel(), height))),
        np.column_stack((X.ravel(), np.full_like(Y.ravel(), - length / 2), Z.ravel())),
        np.column_stack((X.ravel(), np.full_like(Y.ravel(), length / 2), Z.ravel())),
        np.column_stack((np.full_like(X.ravel(), - width / 2), Y.ravel(), Z.ravel())),
        np.column_stack((np.full_like(X.ravel(), width / 2), Y.ravel(), Z.ravel()))
    ])
    
    points_xy = points[:, :2]

    rotated_point = _rotate_vecter(vector=points_xy,
                                   yaw=random.uniform(-math.pi, math.pi),
                                   dimension=2
                                   )
    
    rotated_point = rotated_point + center

    points = np.hstack((rotated_point, points[:, 2:3]))

    return points

def _generate_multiple_boxes(num_boxes,
                             grid_size,
                             point_density
                             ):

    all_points = []

    positions = []
    max_attempts = 1000
    attempts = 0

    while len(positions) < num_boxes and attempts < max_attempts:

        new_position = np.random.uniform(-(grid_size/ 2 - 1.5), grid_size / 2 - 1.5, 2)

        width = np.random.uniform(0.2, 2.0)
        length = np.random.uniform(0.2, 2.0)
        height = np.random.uniform(0.08, 0.25)
        
        intersects = False
        for pos in positions:
            distance = np.sqrt((new_position[0] - pos[0])**2 + (new_position[1] - pos[1])**2)
            if distance < 3:
                intersects = True
                break
        
        if not intersects:
            positions.append(new_position)

            points = _generate_box_point_cloud(new_position,
                                               width,
                                               length,
                                               height,
                                               point_density,
                                               mode=1
                                               )
            all_points.append(points)
        
        attempts += 1

    return np.vstack(all_points)

def _generate_multiple_pillars(num_pillars,
                               grid_size,
                               point_density
                               ):
    all_points = []

    positions = []
    max_attempts = 1000
    attempts = 0

    while len(positions) < num_pillars and attempts < max_attempts:

        new_position = np.random.uniform(-(grid_size/ 2 - 1), grid_size / 2 - 1, 2)

        width = np.random.uniform(0.2, 1)
        length = np.random.uniform(0.2, 1)
        height = 4
        
        intersects = False
        for pos in positions:
            distance = np.sqrt((new_position[0] - pos[0])**2 + (new_position[1] - pos[1])**2)
            if distance < 3:
                intersects = True
                break
        
        if not intersects:
            positions.append(new_position)

            points = _generate_box_point_cloud(new_position,
                                               width,
                                               length,
                                               height,
                                               point_density,
                                               mode=2
                                               )
            all_points.append(points)
        
        attempts += 1

    return np.vstack(all_points)

def _generate_multiple_walls(num_walls,
                             grid_size,
                             point_density
                             ):

    all_points = []

    positions = []
    max_attempts = 1000
    attempts = 0

    while len(positions) < num_walls and attempts < max_attempts:

        new_position = np.random.uniform(-(grid_size/ 2 - 5), grid_size / 2 - 5, 2)

        width = np.random.uniform(0.2, 0.5)
        length = np.random.uniform(5, 8)
        height = 4
        
        intersects = False
        for pos in positions:
            distance = np.sqrt((new_position[0] - pos[0])**2 + (new_position[1] - pos[1])**2)
            if distance < 8:
                intersects = True
                break
        
        if not intersects:
            positions.append(new_position)

            points = _generate_box_point_cloud(new_position,
                                               width,
                                               length,
                                               height,
                                               point_density,
                                               mode=3
                                               )
            all_points.append(points)
        
        attempts += 1

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

def _rotate_vecter(vector,
                   yaw, 
                   dimension
                   ):

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

def _check_vaildation(point_cloud,
                      robot_position,
                      robot_size,
                      detection_range
                      ):

    x_min, x_max = robot_position[0] - robot_size[0] / 1.5, robot_position[0] + robot_size[0] / 1.5
    y_min, y_max = robot_position[1] - robot_size[1] / 1.5, robot_position[1] + robot_size[1] / 1.5
    z_min, z_max = detection_range / 8, detection_range

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
                         robot_size, 
                         detection_range,
                         add_clusters=False
                         ):

    POS_NOISE_RANGE = 0.05
    TILT_ANGLE_RANGE = 1
    HEIGHT_NOISE_RANGE = 0.05
    PRUNING_PERCENTAGE = random.uniform(0.05, 0.1)
    NUM_CLUSTERS = random.randint(4, 7)
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

    if add_clusters:
        cluster_center_range = [[- detection_range / 2.5, detection_range / 2.5],
                                [- detection_range / 2.5, detection_range / 2.5],
                                [0, detection_range / 2]]

        num_clusters = 0
        while(num_clusters < NUM_CLUSTERS):
            cluster_center = np.array([random.uniform(cluster_center_range[0][0], cluster_center_range[0][1]),
                                    random.uniform(cluster_center_range[1][0], cluster_center_range[1][1]),
                                    random.uniform(cluster_center_range[2][0], cluster_center_range[2][1])])
            if _check_vaildation(cluster_center,
                                np.array([0, 0, 0]),
                                robot_size,
                                detection_range
                                ):
                continue

            cluster_points = cluster_center + np.random.normal(scale=0.08, size=(POINTS_PER_CLUSER, 3))
            point_cloud = np.vstack((point_cloud, cluster_points))
            num_clusters += 1
        
    return point_cloud

class TerrainGenerator():

    @staticmethod
    def generate_environment(env_config,
                             visualize=False
                             ):

        grid_size = env_config['grid_size']
        num_obstacles = env_config['num_obstacles']
        point_density = env_config['point_density']

        ground = _generate_ground(grid_size, point_density)
        environment = ground

        if num_obstacles['num_boxes'] != 0:
            boxes = _generate_multiple_boxes(num_obstacles['num_boxes'], grid_size, point_density)
            environment = np.vstack([environment, boxes])

        if num_obstacles['num_pillars'] != 0:
            pillars = _generate_multiple_pillars(num_obstacles['num_pillars'], grid_size, point_density)
            environment = np.vstack([environment, pillars])

        if num_obstacles['num_walls'] != 0:
            walls = _generate_multiple_walls(num_obstacles['num_walls'], grid_size, point_density)
            environment = np.vstack([environment, walls])

        if visualize:
            environment = np.vstack([environment, np.array([0, 0, grid_size])])

        return environment
    
    @staticmethod
    def generate_env_configs(grid_size,
                             point_density,
                             num_boxes_range,
                             num_pillars_range,
                             num_walls_range,
                             num_env_configs
                             ):

        env_configs = []

        for i in range(num_env_configs):
            num_boxes = random.randint(num_boxes_range[0], num_boxes_range[1])
            num_pillars = random.randint(num_pillars_range[0], num_pillars_range[1])
            num_walls = random.randint(num_walls_range[0], num_walls_range[1])
            env_config = {'grid_size': grid_size, 
                          'num_obstacles': {'num_boxes': num_boxes, 
                                            'num_pillars': num_pillars, 
                                            'num_walls': num_walls},
                          'point_density': point_density
                          }
            env_configs.append(env_config)
        
        return env_configs
    
    @staticmethod
    def generate_robot_configs(grid_size, 
                               detection_range, 
                               robot_size, 
                               robot_speed, 
                               num_time_steps,
                               time_step
                               ):
        if time_step != None:
            range_value = grid_size / 2 - detection_range / 2 - num_time_steps*time_step*robot_speed
        else:
            range_value = grid_size / 2 - detection_range / 2

        robot_positions = []
        robot_yaws = []
        for i in range(num_time_steps + 1):
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
                robot_yaw = robot_yaw + random.uniform(-math.pi / 6, math.pi / 6)
                    
                robot_positions.append(robot_position)
                robot_yaws.append(robot_yaw)

        return robot_positions, robot_yaws

    @staticmethod
    def filter_points_in_detection_area(environment,
                                        detection_range,
                                        robot_size,
                                        robot_position,
                                        visualize=False
                                        ):
            
        if _check_vaildation(environment,
                             robot_position,
                             robot_size,
                             detection_range
                             ):
            return None

        translated_points = environment[:, :2] - robot_position[:2]
        translated_points = np.hstack([translated_points, environment[:, 2:3]])
            
        x_min = -detection_range / 2
        x_max = detection_range / 2
        y_min = -detection_range / 2
        y_max = detection_range / 2
        z_min = 0
        z_max = 3.2

        filtered_points = translated_points[
            (translated_points[:, 0] >= x_min) & (translated_points[:, 0] <= x_max) &
            (translated_points[:, 1] >= y_min) & (translated_points[:, 1] <= y_max) &
            (translated_points[:, 2] >= z_min) & (translated_points[:, 2] <= z_max)
        ]
            
        if visualize:
            filtered_points = np.vstack([filtered_points,
                                        np.array([0,
                                                  0,
                                                  detection_range])])
                            
        return filtered_points

    @staticmethod
    def senser_detection(point_cloud,
                         detection_range,
                         robot_size,
                         sensor_config,
                         visualize=False
                         ):
        
        sensors = _generate_sensors(sensor_config['tilt_angle'], 
                                    sensor_config['fov_angle'], 
                                    sensor_config['detection_distance'], 
                                    sensor_config['relative_position']
                                    )

        robot_position = np.array([random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), robot_size[2]])

        detected_points = []
            
        for point in point_cloud:
            visible = False
            for sensor in sensors:
                sensor_position = np.array(sensor['position']) + robot_position
                if _is_in_fov(point, 
                              sensor_position,
                              np.array(sensor['direction']),
                              sensor['fov'],
                              sensor['max_distance']
                              ):
                    visible = True
                    break

            if visible:
                detected_points.append(point)

        detected_points = np.array(detected_points)
            
        detected_points = _noisify_point_cloud(detected_points,
                                               robot_size,
                                               detection_range
                                               )
        
        if visualize:
            detected_points = np.vstack([detected_points,
                                         np.array([0, 0, detection_range])])
        
        return detected_points
    
    @staticmethod
    def voxelize_pc(point_cloud,
                    voxel_resolution,
                    time_index
                    ):

        points = point_cloud
        min_coord = points.min(axis=0)
        width = points[:, 0].max() - points[:, 0].min()
        depth = points[:, 1].max() - points[:, 1].min()
        height = points[:, 2].max() - points[:, 2].min()
        detection_range = max(width, depth, height)
        points_normalized = (points - min_coord) / (detection_range + 1e-15)
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