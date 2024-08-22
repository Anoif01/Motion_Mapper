# Motion Mapper: Pose Estimation and Trajectory Tracking

[![Demo Video](https://github.com/Anoif01/Motion_Mapper/assets/results/output_football_player1.gif)]  
**Click the image above to view the demo on Hugging Face Spaces**

---

## Overview ⚡⚡⚡
**Motion Mapper** is a comprehensive tool for pose estimation, trajectory tracking, and movement heatmap visualization in **fixed-angle** sports videos. 
This tool leverages the power of the **YOLOv8 pose series models** to detect and analyze human poses, visualize trajectories, and generate heatmaps to highlight areas of intense activity.

## Key Features 🌟🌟🌟
- **Pose Estimation**: Accurate human pose detection using YOLOv8 pose models.
- **Trajectory Tracking**: Real-time tracking of movement trajectories for multiple subjects.
- **Heatmap Visualization**: Dynamic heatmaps that visualize areas with high activity levels, helping to identify patterns and intensities of movements.
- **Model Flexibility**: Supports all models in the YOLOv8 pose series for varying levels of performance and accuracy.
    - yolov8n-pose.pt
    - yolov8s-pose.pt
    - yolov8m-pose.pt
    - yolov8l-pose.pt
    - yolov8x-pose.pt

## Hugging Face Space Demo 🔥🔥🔥
[Check out the live demo on Hugging Face Spaces](https://huggingface.co/spaces/HappyOtter/MotionMapper)

## Getting Started

### Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Anoif01/Motion_Mapper.git
    cd Motion_Mapper
    ```

2. **Install required dependencies**:

    Make sure you have a virtual python environment activated, then run:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up configuration**:

    Open the `src/config.py` file and set the following variables according to your needs. Below are the settings that must be changed:

    ```python
    # Paths to input and output files
    video_path = "/assets/exemple.mp4"  # Change to your input video path
    resize_video_path = "/assets/resize_exemple.mp4"  # Change to your resized fps video path
    target_fps = 10  # Target frames per second for processing
    output_video_path = "/assets/results/output_exemple.mp4"  # Change to your output video path

    # YOLO model configuration
    model_name = 'yolov8m-pose.pt'  # Choose your model
    ```

4. **Place your video in the `/assets/` directory**:

    Make sure your video file is placed in the `/assets/` directory. The filename should match the `video_path` specified in `config.py`.

### Running the Project

After setting up the configuration and placing your video file in the appropriate directory, you can process the video by running:

```bash
python main.py
```

The processed video, along with trajectories and heatmaps, will be saved to the location specified in output_video_path.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8) for the powerful pose estimation models.
- [Streamlit](https://www.streamlit.io/) for the interactive UI framework.
- [Hugging Face](https://www.huggingface.co/) for providing an excellent platform for hosting ML models and applications.

