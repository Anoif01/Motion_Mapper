# -*- coding: utf-8 -*-
"""
source/streamlit_app.py

This is the streamlit user interface.

Author: xiaohua.lu
Date Created: 2024/8/20
"""
import streamlit as st
import subprocess
import os
import cv2
from main import main
import config


# Streamlit app title
st.title("Motion Mapper： a pose estimation and trajectory tracking app")

UPLOAD_DIRECTORY = "/assets/"

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Start button
if st.button("Start Processing"):
    if uploaded_file is not None:
        if not os.path.exists(UPLOAD_DIRECTORY):
            os.makedirs(UPLOAD_DIRECTORY)

        filename = uploaded_file.name
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # update config
        config.video_path = file_path
        output_video_path = os.path.join(UPLOAD_DIRECTORY, 'results', 'output_'+filename)
        config.output_video_path = output_video_path

        # Display a processing message
        st.write("Processing... Please wait, this may take a while.")

        # Run the main function with the updated config
        main(config)
        
        # ffmpeg command convert to H.264
        h264_output_video_path = "output_h264_video.mp4"
        ffmpeg_command = f"ffmpeg -i {output_video_path} -vcodec libx264 {h264_output_video_path}"

        # execute ffmpeg
        subprocess.run(ffmpeg_command, shell=True)

        # Display the processed video
        if os.path.exists(h264_output_video_path):
            st.write("Processing complete! Here is the processed video:")
            st.video(h264_output_video_path)
        else:
            st.error("Failed to convert video to H.264 format.")
    else:
        st.write("Please upload a video file to start processing.")