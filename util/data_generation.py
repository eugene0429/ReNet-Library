import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import random
from util.data_processing import DataProcessing

class DataGeneration():
    def generate_box_point_cloud(self, center, width, length, height, point_density):

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

    def generate_multiple_boxes(self, num_boxes, grid_size, point_density):

        all_points = []

        for _ in range(num_boxes):
            width = np.random.uniform(0.2, 2.0)
            length = np.random.uniform(0.2, 2.0)
            height = np.random.uniform(0.08, 0.25)

            center = np.random.uniform(1, grid_size - 1, 2)

            points = self.generate_box_point_cloud(center, width, length, height, point_density)

            all_points.append(points)

        return np.vstack(all_points)

    def generate_multiple_pillars(self, num_pillars, grid_size, point_density):

        all_points = []

        for _ in range(num_pillars):
            width = np.random.uniform(0.2, 1)
            length = np.random.uniform(0.2, 1)
            height = 4

            center = np.random.uniform(1, grid_size - 1, 2)

            points = self.generate_box_point_cloud(center, width, length, height, point_density)

            all_points.append(points)

        return np.vstack(all_points)

    def generate_multiple_walls(self, num_walls, grid_size, point_density):

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

            center = np.random.uniform(5, grid_size - 5, 2)

            points = self.generate_box_point_cloud(center, width, length, height, point_density)

            all_points.append(points)

        return np.vstack(all_points)

    def generate_ground(self, grid_size, point_density):
        
        width = grid_size
        length = grid_size

        center = np.array([grid_size/2, grid_size/2])

        x = np.linspace(center[0] - width / 2, center[0] + width / 2, int(point_density * width))
        y = np.linspace(center[1] - length / 2, center[1] + length / 2, int(point_density * length))

        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X.ravel())

        points = np.column_stack((X.ravel(), Y.ravel(), Z))
        points = np.row_stack((points, np.array([grid_size, grid_size, grid_size])))
        
        return points

    def generate_environment(self, grid_size, num_boxes, num_pillars, num_walls, point_density):

        ground = self.generate_ground(grid_size, point_density)
        boxes = self.generate_multiple_boxes(num_boxes, grid_size, point_density)
        pillars = self.generate_multiple_pillars(num_pillars, grid_size, point_density)
        walls = self.generate_multiple_walls(num_walls, grid_size, point_density)

        return np.vstack([ground, boxes, pillars, walls])

    def filter_points_in_detection_area(self, point_cloud, robot_position, detection_range=3.2):
        x_min = robot_position[0] - detection_range / 2
        x_max = robot_position[0] + detection_range / 2
        y_min = robot_position[1] - detection_range / 2
        y_max = robot_position[1] + detection_range / 2
        z_min = 0
        z_max = 3.2

        filtered_points = point_cloud[
            (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) &
            (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max) &
            (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] <= z_max)
        ]
        
        return filtered_points


DP = DataProcessing()
DG = DataGeneration()

grid_size = 20
num_boxes = 20
num_pillars = 10
num_walls = 5
point_density = 15
robot_height = 1.2

robot_position = np.array([random.uniform(0, grid_size - 3.2), random.uniform(0, grid_size - 3.2), robot_height])

environment = DG.generate_environment(grid_size, num_boxes, num_pillars, num_walls, point_density)

data = DG.filter_points_in_detection_area(environment, robot_position)

DP.visualize_pc(data)

