"""
config.py

This module stores configuration settings such as file paths, model parameters, and other
constants used across the project.

Author: xiaohua.lu
Date Created: 2024/8/20
"""

# Paths to input and output files
video_path = "./assets/football_player1.mp4"
resize_video_path = "./assets/resize_football_player1.mp4"
target_fps = 10
output_video_path = "./assets/results/output_football_player1.mp4"

# YOLO model configuration, chek for more options https://docs.ultralytics.com/tasks/pose/?h=pose
model_name = 'yolov8m-pose.pt'

# Video processing parameters
resize_shape = (500, 380)  # (width, height)

# Draw skeleton settings
line_thickness = 15  # Thickness of the skeleton lines
trans_line_thickness = 10  # Thickness of the transparent overlay lines
scatter_radius = 15  # Radius of the scatter points
line_color = (241, 200, 12)  # Radius of the scatter points
trans_line_color = (241, 225, 149)  # Radius of the scatter points
scatter_color = (32, 32, 244)  # Radius of the scatter points
alpha_overlay = 0.65  # Transparency level of the overlay

# Draw trajectory settings
# trajectory_thickness = 100  # Thickness of the trajectory lines
# trajectory_alpha = 0.65  # Transparency level of the overlay
traj_radius = 50  # Radius of the trajectory circles

# Draw heatmap settings
heatmap_alpha = 0.35  # Transparency of heatmap overlay
