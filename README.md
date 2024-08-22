# Motion_Mapper
A pose estimation and trajectory tracking streanlit application
Motion Mapper: Pose Estimation and Trajectory Tracking

Click the image above to view the demo on Hugging Face Spaces

Overview
Motion Mapper is a comprehensive tool for pose estimation, trajectory tracking, and movement heatmap visualization in fixed-angle sports videos. This tool leverages the power of the YOLOv8 pose series models to detect and analyze human poses in real-time, visualize trajectories, and generate heatmaps to highlight areas of intense activity.

Key Features
Pose Estimation: Accurate human pose detection using YOLOv8 pose models.
Trajectory Tracking: Real-time tracking of movement trajectories for multiple subjects.
Heatmap Visualization: Dynamic heatmaps that visualize areas with high activity levels, helping to identify patterns and intensities of movements.
Model Flexibility: Supports all models in the YOLOv8 pose series for varying levels of performance and accuracy.
Demo
Check out the live demo on Hugging Face Spaces

Supported Models
This project supports the following YOLOv8 pose models:

yolov8n-pose.pt
yolov8s-pose.pt
yolov8m-pose.pt
yolov8l-pose.pt
yolov8x-pose.pt
Getting Started
Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
Python 3.8 or higher
git installed on your machine
A video file you want to process, placed in the assets/ directory.
Installation
Clone the repository:

bash
Copier le code
git clone https://github.com/YourUsername/Motion_Mapper.git
cd Motion_Mapper
Install required dependencies:

Make sure you have a virtual environment activated, then run:

bash
Copier le code
pip install -r requirements.txt
Set up configuration:

Open the src/config.py file and set the following variables according to your needs:

python
Copier le code
# Paths to input and output files
video_path = "/assets/exe2.mp4"  # Input video path
resize_video_path = "/assets/resize_exe2.mp4"  # Resized video path
target_fps = 10  # Target frames per second for processing
output_video_path = "/assets/results/yolov8m_exe2.mp4"  # Output video path

# YOLO model configuration
model_name = 'yolov8m-pose.pt'  # Choose your model

# Video processing parameters
resize_shape = (500, 380)  # Resize dimensions (width, height)

# Visualization settings
line_thickness = 15
scatter_radius = 15
alpha_overlay = 0.65

# Heatmap settings
heatmap_alpha = 0.35
Place your video in the assets/ directory:

Make sure your video file is placed in the assets/ directory. The filename should match the video_path specified in config.py.

Running the Project
After setting up the configuration and placing your video file in the appropriate directory, you can process the video by running:

bash
Copier le code
python src/main.py
The processed video, along with trajectories and heatmaps, will be saved to the location specified in output_video_path.

Example Output
Here’s an example of what the processed video looks like:



Hugging Face Space
You can also test the project directly in the browser by visiting the Hugging Face Space.

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Ultralytics YOLOv8 for the powerful pose estimation models.
Streamlit for the interactive UI framework.
Hugging Face for providing an excellent platform for hosting ML models and applications.
By following this guide, you should be able to get your project up and running quickly. If you encounter any issues, please refer to the issues section of the repository or reach out for support.
