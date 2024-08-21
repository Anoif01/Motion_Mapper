"""
src/pose_estimation.py

This module provides functions to get unioque skeletons and draw skeletons on images.

Author: xiaohua.lu

Date Created: 2024/8/20
"""

import cv2
import numpy as np
import math
from utils import connections, keypoint_dict

def calculate_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - x1, y1: Coordinates of the first point.
    - x2, y2: Coordinates of the second point.

    Returns:
    - distance: Euclidean distance between the two points.
    """
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def find_unique_points(A, epsilon):
    """
    Find unique points in list A, returning their indices.

    Parameters:
    - A: List of points.
    - epsilon: Minimum distance to consider points as unique.

    Returns:
    - unique_A: List of unique points.
    - unique_indices: Indices of unique points in A.
    """
    unique_A = []
    unique_indices = []

    for i, point in enumerate(A):
        is_unique = True
        for unique_point in unique_A:
            if calculate_distance(point[0], point[1], unique_point[0], unique_point[1]) < epsilon:
                is_unique = False
                # print(calculate_distance(point[0], point[1], unique_point[0], unique_point[1]))
                break

        if is_unique:
            unique_A.append(point)
            unique_indices.append(i)

    return unique_A, unique_indices

def check_skeletons_in_frame(results):
    """
    Check for skeletons in the current frame and return unique skeletons.

    Parameters:
    - results: YOLO results containing keypoint data.

    Returns:
    - unique_skeletons: List of unique skeleton centers.
    - unique_skeleton_indices: Indices of the unique skeletons.
    """
    skeletons = []
    for id_person, res in enumerate(results[0]):
        result_keypoint = res.keypoints.xy.cpu().numpy()[0]
        x_left_hip, y_left_hip = result_keypoint[11]
        x_right_hip, y_right_hip = result_keypoint[12]

        x_hip_center = (x_left_hip + x_right_hip) / 2.0
        y_hip_center = (y_left_hip + y_right_hip) / 2.0

        skeletons.append([x_hip_center, y_hip_center])

    unique_skeletons, unique_skeleton_indices = find_unique_points(skeletons, epsilon=10)
    return unique_skeletons, unique_skeleton_indices

def draw_skeleton(yolo_results, image, line_thickness=15, trans_line_thickness=40, 
                  scatter_thickness=-1, line_color=(241, 200, 12), 
                  trans_line_color=(241, 225, 149), scatter_radius=15, 
                  scatter_color=(32, 32, 244), alpha_overlay=0.5):
    """
    Draw skeletons on the given image based on YOLO pose estimation results.

    Parameters:
    - yolo_results: Results from YOLO pose estimation.
    - image: The image on which skeletons are to be drawn.
    - line_thickness: Thickness of the main skeleton lines.
    - trans_line_thickness: Thickness of the transparent overlay lines.
    - scatter_thickness: Thickness of the scatter points (key points).
    - line_color: Color of the main skeleton lines (BGR format).
    - trans_line_color: Color of the transparent overlay lines (BGR format).
    - scatter_radius: Radius of the scatter points.
    - scatter_color: Color of the scatter points (BGR format).
    - alpha_overlay: Transparency level of the overlay.

    Returns:
    - image: The image with skeletons drawn on it.
    """

    # Create overlay layers
    overlay = image.copy()
    original = image.copy()

    for res in yolo_results[0]:
        result_keypoint = res.keypoints.xy.cpu().numpy()[0]

        for connection in connections:
            pt1 = tuple(result_keypoint[keypoint_dict[connection[0]]].astype(int))
            pt2 = tuple(result_keypoint[keypoint_dict[connection[1]]].astype(int))

            cv2.line(overlay, pt1, pt2, color=line_color, thickness=trans_line_thickness)
            cv2.line(original, pt1, pt2, color=line_color, thickness=line_thickness)

        for i, (x, y) in enumerate(result_keypoint):
            cv2.circle(original, (int(x), int(y)), radius=scatter_radius, 
                       color=scatter_color, thickness=scatter_thickness)

    image = cv2.addWeighted(original, 0.9, image, 0.1, 0)
    image = cv2.addWeighted(overlay, alpha_overlay, image, 1 - alpha_overlay, 0)

    return image
