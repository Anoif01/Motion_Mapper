"""
source/app.py

This is the main application file that integrates pose estimation, trajectory tracking,
and heatmap generation, and processes video input to generate annotated video output.

Author: xiaohua.lu
Date Created: 2024/8/20
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pose_estimation import calculate_distance, check_skeletons_in_frame, draw_skeleton
from trajectory_tracking import  remove_outliers, gaussian_smooth, update_trajectories, draw_gradient_line
from heatmap_generation import calculate_speeds, create_global_speed_matrix, generate_heatmap, overlay_heatmap
from utils import generate_colors, keypoint_dict, connections, reduce_frame_rate
#from config import *
import config


def main(cfg):
    """
    Main function to process video input and generate annotated video output.

    Parameters:
    - cfg: Configuration module containing settings like paths, model parameters, etc.
    """

    # =============================================================================
    #     Model initialization
    # =============================================================================

    model = YOLO(cfg.model_name)

    # =============================================================================
    #     Video capture and output settings
    # =============================================================================
    
    resize_video_path = reduce_frame_rate(cfg.video_path, cfg.resize_video_path, cfg.target_fps)
    
    cap = cv2.VideoCapture(cfg.resize_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_time = 1.0 / fps
    
    if cfg.video_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif cfg.video_path.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif cfg.video_path.endswith('.mov'):
        fourcc = cv2.VideoWriter_fourcc(*'H264')
    else:
        raise ValueError("Unsupported file format: {}".format(cfg.resize_video_path))

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_output_size = (frame_width, frame_height)
    out = cv2.VideoWriter(cfg.output_video_path, fourcc, fps, save_output_size)
    
    # =============================================================================
    #     main function
    # =============================================================================
    # Trajectories storage
    trajectories = {}
    frame_id = 0

    # Process video frames
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            frame_id += 1
    
            # Run YOLOv8 inference on the frame
            results = model(frame)
    
            # Draw skeletons on the frame
            annotated_frame = draw_skeleton(results, frame, 
                                            line_thickness=cfg.line_thickness, 
                                            trans_line_thickness=cfg.trans_line_thickness, 
                                            scatter_radius=cfg.scatter_radius, 
                                            line_color=cfg.line_color,
                                            trans_line_color=cfg.trans_line_color,
                                            scatter_color=cfg.scatter_color,
                                            alpha_overlay=cfg.alpha_overlay)
    
            # Remove duplicate skeletons and track unique ones
            unique_skeletons, unique_skeleton_indices = check_skeletons_in_frame(results)
            person_indices = [pi for pi in range(len(unique_skeleton_indices))]
    
            # Update trajectories using the new function
            trajectories = update_trajectories(trajectories, unique_skeletons, unique_skeleton_indices, person_indices, results, frame_id)
    
    
            # Draw trajectories
            bgr_colors = generate_colors(len(trajectories))
            for object_id, points in trajectories.items():
                points = remove_outliers(points)
                smoothed_points = gaussian_smooth(points, sigma=1)
                start_pt = tuple([int(smoothed_points[0]),int(smoothed_points[2]+100)])
                end_pt = tuple([int(smoothed_points[-3]),int(smoothed_points[-1])+100])
                color1 = bgr_colors[object_id]
                color2= (255 , 255 , 255)
                # cv2.line(annotated_frame, start_pt, end_pt, color, thickness=80)
                annotated_frame = draw_gradient_line(annotated_frame, start_pt, end_pt, color2, color1, cfg.trajectory_thickness, cfg.trajectory_alpha)
    
            # Generate and overlay heatmap
            img_shape = annotated_frame.shape[:2]
            speeds = calculate_speeds(trajectories, frame_time)
            global_speed_matrix = create_global_speed_matrix(speeds, img_shape)
            heatmap = generate_heatmap(global_speed_matrix, img_shape)
            annotated_frame = overlay_heatmap(annotated_frame, heatmap, alpha=cfg.heatmap_alpha)
    
            # Resize and save the frame
            # annotated_frame_resized = cv2.resize(annotated_frame, cfg.resize_shape)
            # Display the annotated frame
            # cv2_imshow(annotated_frame_resized)
        
            out.write(annotated_frame)
        else:
          # Break the loop if the end of the video is reached
          break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(config)  # Pass the config module as an argument to the main function