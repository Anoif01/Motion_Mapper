"""
src/heatmap_generation.py

This module provides functions to calculate speeds based on object trajectories, create a global speed matrix, and generate heatmaps.

Author: xiaohua.lu
Date Created: 2024/8/20
"""

import cv2
import numpy as np
from src.pose_estimation import calculate_distance

def calculate_speeds(trajectories, frame_time):
    """
    Calculate speeds based on trajectories and frame time.
    
    Parameters:
    - trajectories: Dictionary with person IDs as keys and list of points as values.
    - frame_time: Time duration of each frame.
    Returns:
    - speeds: Dictionary with person IDs as keys and list of (x, y, y_ankle, speed) tuples as values.
    """
    speeds = {}
    for person_id, points in trajectories.items():
        points_x= np.array([points[p] for p in range(0, len(points), 3) if points[p]>0])
        points_y= np.array([points[p] for p in range(1, len(points), 3) if points[p]>0])
        points_y_ankle= np.array([points[p] for p in range(2, len(points), 3) if points[p]>0])

        speeds[person_id] = []
        for i in range(1, len(points_x)):
            dist = calculate_distance(points_x[i], points_y[i], points_x[i-1], points_y[i-1])
            speed = dist / frame_time
            speeds[person_id] += [(points_x[i], points_y[i], points_y_ankle[i], speed)]
    return speeds

def create_global_speed_matrix(speeds, img_shape):
    """
    Create a global speed matrix based on the speeds and image shape.
    
    Parameters:
    - speeds: Dictionary with person IDs as keys and list of (x, y, y_ankle, speed) tuples as values.
    - img_shape: Shape of the image (height, width).
    Returns:
    - speed_matrix: 2D numpy array representing the speed matrix.
    """
    speed_matrix = np.zeros((img_shape[0], img_shape[1]), dtype=np.float32)

    for person_id, speed_data in speeds.items():
        try:
            point_x, point_y, point_y_ankle, speed = speed_data[-1]

            # Get the range of x and y
            x_min = max(0, int(point_x) - 200)
            x_max = min(img_shape[1] - 1, int(point_x) + 200)
            y_min = max(0, int(point_y_ankle)-2*(int(point_y_ankle)-int(point_y)))
            y_max = min(img_shape[0] - 1, int(point_y_ankle))

            speed_matrix[y_min:y_max+1, x_min:x_max+1] += speed

        except:
            pass

    return speed_matrix

def generate_heatmap(speed_matrix, img_shape):
    """
    Generate a heatmap from the speed matrix.

    Parameters:
    - speed_matrix: 2D numpy array representing the speed matrix.
    - img_shape: Shape of the image (height, width).

    Returns:
    - heatmap: Heatmap image generated from the speed matrix.
    """
    normalized_speed_matrix = cv2.normalize(speed_matrix, None, 0, 255, cv2.NORM_MINMAX)
    normalized_speed_matrix = np.uint8(normalized_speed_matrix)
    heatmap = cv2.applyColorMap(normalized_speed_matrix, cv2.COLORMAP_JET)
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.6):
    """
    Overlay a heatmap onto the original image.

    Parameters:
    - image: Original image.
    - heatmap: Heatmap to overlay.
    - alpha: Transparency factor for overlay.

    Returns:
    - overlayed_image: Image with heatmap overlayed.
    """
    overlay = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
    return overlay
