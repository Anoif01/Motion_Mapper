"""
src/utils.py

This module includes utility functions such as color generation for visualization, and constants like keypoint mappings and connections for pose estimation.

Author: xiaohua.lu
Date Created: 2024/8/20
"""

import matplotlib.pyplot as plt
import cv2

def reduce_frame_rate(video_path, output_video_path, target_fps):
    """
    Reduce the frame rate of a video to speed up processing.

    Parameters:
    - input_video_path: Path to the input video.
    - output_video_path: Path to save the video with reduced frame rate.
    - target_fps: The desired frames per second (FPS) for the output video.

    Returns:
    - output_video_path: Path to the saved video with reduced frame rate.
    """
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps
    
    # Calculate the interval to drop frames
    frame_interval = int(original_fps / target_fps)
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    if video_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif video_path.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif video_path.endswith('.mov'):
        fourcc = cv2.VideoWriter_fourcc(*'H264')
    else:
        raise ValueError("Unsupported file format: {}".format(video_path))
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))

    # Process and write frames
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Write only every nth frame
        if frame_id % frame_interval == 0:
            out.write(frame)
        
        frame_id += 1

    cap.release()
    out.release()
    print('Video fps is reseted.')
    
    return output_video_path


def check_negative_in_list(A):
    """
    Count the number of negative elements in a list. 

    Parameters:
    - A: List of points.

    Returns:
    - count_negative: Number of negative elements.
    """
    negative_numbers = list(filter(lambda x: x < 0, A))
    count_negative = 0

    # 判断是否有小于0的数并计算数量
    if negative_numbers:
        count_negative = len(negative_numbers)
    
    return count_negative

# Keypoint dictionary mapping for human pose estimation
keypoint_dict = {
    "NOSE": 0, "LEFT_EYE": 1, "RIGHT_EYE": 2, "LEFT_EAR": 3, "RIGHT_EAR": 4,
    "LEFT_SHOULDER": 5, "RIGHT_SHOULDER": 6, "LEFT_ELBOW": 7, "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9, "RIGHT_WRIST": 10, "LEFT_HIP": 11, "RIGHT_HIP": 12,
    "LEFT_KNEE": 13, "RIGHT_KNEE": 14, "LEFT_ANKLE": 15, "RIGHT_ANKLE": 16,
}

# Connection pairs defining the human skeleton
connections = [
    ("NOSE", "LEFT_EYE"), ("NOSE", "RIGHT_EYE"),
    ("LEFT_EYE", "LEFT_EAR"), ("RIGHT_EYE", "RIGHT_EAR"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"), ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"), ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE")
]
def generate_colors(num_colors):
    """
    Generate a list of different colors.

    Parameters:
    - num_colors: Number of colors to generate.

    Returns:
    - bgr_color: List of colors in BGR format.
    """
    rgba_cmap = plt.cm.get_cmap('rainbow', 10)
    rgba_cmap_list = [rgba_cmap(i) for i in range(num_colors)]
    bgr_color = [(int(rgba[2] * 255),  int(rgba[1] * 255), int(rgba[0] * 255)) for rgba in rgba_cmap_list]
    return bgr_color