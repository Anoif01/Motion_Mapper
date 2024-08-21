"""
src/trajectory_tracking.py

This module contains functions for tracking trajectories of detected objects, 
removing outliers, and smoothing the trajectories.

Author: xiaohua.lu
Date Created: 2024/8/20
"""


import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from pose_estimation import calculate_distance


def update_trajectories(trajectories, unique_skeletons, unique_skeleton_indices, person_indices, results, frame_id):
    """
    Update the trajectories based on the current frame's pose estimation results.

    Parameters:
    - trajectories: Dictionary storing the trajectories of each detected person.
    - unique_skeletons: List of unique skeleton center points detected in the current frame.
    - unique_skeleton_indices: Indices of unique skeletons in the current frame's results.
    - person_indices: Indices of persons corresponding to the unique skeletons.
    - results: YOLO model results containing keypoint information.
    - frame_id: The current frame index.

    Returns:
    - trajectories: Updated trajectories dictionary.
    """
    for id_skeleton, id_person, skeleton in zip(unique_skeleton_indices, person_indices, unique_skeletons):
        result_keypoint = results[0][id_skeleton].keypoints.xy.cpu().numpy()[0]
        x_right_ankle, y_right_ankle = result_keypoint[16]
        x_left_ankle, y_left_ankle = result_keypoint[15]
        x_ankle_center = (x_left_ankle + x_right_ankle) / 2.0
        y_ankle_center = (y_left_ankle + y_right_ankle) / 2.0
    
        x_hip_center, y_hip_center = skeleton
    
        # Initialising the trajectories of detected person in the first frame
        if id_person not in trajectories.keys() and frame_id == 1 :
            # print('Initialize !!!')
            trajectories[id_person] = [x_hip_center, y_hip_center, y_ankle_center]
        # Update trajectories
        else:
            min_dist, min_id = np.inf, np.inf
            pos = min([ len(trajectories[l]) for l in range(len(trajectories))])
            # print(pos)
    
            # No additional person detected
            if len(unique_skeletons) <= len(trajectories):
                # print('Less than or Same Person!!!')
                for p in range(len(trajectories)):
                    previous_x, previous_y, previous_yankel = trajectories[p][pos-3 : pos]
                    dist = calculate_distance(previous_x, previous_y, x_hip_center, y_hip_center)
                    # print(dist)
                    if dist < min_dist:
                        min_dist = dist
                        min_id = p
                if min_dist < 500:
                    trajectories[min_id] += [x_hip_center, y_hip_center, y_ankle_center]
                    # print(trajectories)
            # Additional person detected.             
            else:
                print('More Person!!!')
                # diff = len(unique_skeletons) - len(trajectories)
    
                min_dist, min_id = np.inf, 0.
                for p in range(len(trajectories)):
                    previous_x, previous_y, previous_yankel = trajectories[p][pos-3 : pos]
                    dist = calculate_distance(previous_x, previous_y, x_hip_center, y_hip_center)
                    # print(dist)
                    if dist < min_dist:
                        min_dist = dist
                        min_id = p
                if min_dist < 500:
                    trajectories[min_id] += [x_hip_center, y_hip_center, y_ankle_center]
                    # print(trajectories)
                print(id_person)
    
                if id_person >= len(trajectories):
                    # new person detected!
                    print('New person added!')
                    trajectories[id_person] = [x_hip_center, y_hip_center, y_ankle_center]
                        # print(len(trajectories)+1, trajectories[len(trajectories)+1])
                print(trajectories)

    return trajectories

def remove_outliers(points, max_distance=200):
    """
    Remove outlier points from the trajectory.

    Parameters:
    - points: List of points in the trajectory.
    - max_distance: Maximum allowed distance between points.

    Returns:
    - filtered_points: List of points with outliers removed.
    """
    points_x= np.array([points[p] for p in range(0, len(points), 3)])
    points_y= np.array([points[p] for p in range(1, len(points), 3)])
    points_y_ankle= np.array([points[p] for p in range(2, len(points), 3)])

    filtered_points_x = [points_x[0]]  
    filtered_points_y = [points_y[0]]  
    filtered_points = [points_x[0], points_y[0], points_y_ankle[0]]

    for i in range(1, len(points_x)):
        # dist = np.linalg.norm(np.array(points[i]) - np.array(filtered_points[-1]))
        dist = calculate_distance(points_x[i], points_y[i], filtered_points_x[-1], filtered_points_y[-1])
        if dist < max_distance:
            # print(dist)
            filtered_points += [points_x[i], points_y[i], points_y_ankle[i]]
            filtered_points_x.append(points_x[i])
            filtered_points_y.append(points_y[i])
        else:
            print('dist > max_dist: ', dist)
            print(i, [points_x[i], points_y[i], points_y_ankle[i]])
            print(filtered_points_x[-1], filtered_points_y[-1])
    return filtered_points

def gaussian_smooth(points, sigma=1):
    """
    Apply Gaussian smoothing to trajectory points.

    Parameters:
    - points: List of points in the trajectory.
    - sigma: Standard deviation for Gaussian kernel.

    Returns:
    - smoothed_points: List of smoothed points.
    """
    points_np = np.array(points)
    points_x= np.array([points_np[p] for p in range(0, len(points_np), 3)])
    points_y= np.array([points_np[p] for p in range(1, len(points_np), 3)])
    points_y_ankle= np.array([points[p] for p in range(2, len(points), 3)])
    # print(len(points_x), len(points_y), len(points_y_ankle))

    smoothed_x = gaussian_filter1d(points_x, sigma=sigma)
    smoothed_y_ankle = gaussian_filter1d(points_y_ankle, sigma=sigma)
    # print(len(smoothed_x), len(points_y), len(smoothed_y_ankle))

    smoothed_points = [smoothed_x[0], points_y[0], smoothed_y_ankle[0]]
    for i in range(len(smoothed_x)):
        smoothed_points += [smoothed_x[i], points_y[i], smoothed_y_ankle[i]]
    return smoothed_points

def draw_gradient_line(img, pt1, pt2, color1, color2, thickness=2, alpha=0.5):
    """
    Draws a line with a gradient color and transparency on an image.

    Parameters:
    - img: The image on which the line will be drawn.
    - pt1: The starting point of the line (x, y).
    - pt2: The ending point of the line (x, y).
    - color1: The starting color of the gradient (BGR format).
    - color2: The ending color of the gradient (BGR format).
    - thickness: The thickness of the line.
    - alpha: The transparency level of the line.

    Returns:
    - img: The image with the gradient line drawn on it.
    """
    distance = np.linalg.norm(np.array(pt2) - np.array(pt1))

    # Create an overlay image for drawing the gradient line
    overlay = img.copy()

    for i in range(int(distance)):
        t = i / distance
        color = (
            int(color1[0] * (1 - t) + color2[0] * t),
            int(color1[1] * (1 - t) + color2[1] * t),
            int(color1[2] * (1 - t) + color2[2] * t),
        )
        pt_start = (
            int(pt1[0] * (1 - t) + pt2[0] * t),
            int(pt1[1] * (1 - t) + pt2[1] * t),
        )
        pt_end = (
            int(pt1[0] * (1 - (i + 1) / distance) + pt2[0] * (i + 1) / distance),
            int(pt1[1] * (1 - (i + 1) / distance) + pt2[1] * (i + 1) / distance),
        )

        cv2.line(overlay, pt_start, pt_end, color, thickness)


    # Blend the overlay with the original image
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return img