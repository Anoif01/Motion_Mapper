# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 23:34:04 2024

@author: luxia
"""

import os
import cv2

os.chdir('C:\_work\_kaggle\MotionMapper')

# 定义视频和缩略图目录
video_directory = './assets/'
thumbnail_directory = './assets/thumbnails/'

# 创建缩略图目录（如果不存在）
if not os.path.exists(thumbnail_directory):
    os.makedirs(thumbnail_directory)

# 遍历 assets/ 目录下的所有文件
for filename in os.listdir(video_directory):
    if filename.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(video_directory, filename)

        # 使用 cv2 读取视频
        cap = cv2.VideoCapture(video_path)

        # 跳到第五帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5)

        success, frame = cap.read()

        if success:
            # 构建缩略图保存路径
            thumbnail_path = os.path.join(thumbnail_directory, f"{os.path.splitext(filename)[0]}.png")

            height, width = frame.shape[:2]
            max_height = 300
            max_width = 300
            
            # only shrink if img is bigger than required
            if max_height < height or max_width < width:
                # get scaling factor
                scaling_factor = max_height / float(height)
                if max_width/float(width) < scaling_factor:
                    scaling_factor = max_width / float(width)
                # resize image
                frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                
            cv2.imwrite(thumbnail_path, frame)

            print(f"Saved thumbnail for {filename} at {thumbnail_path}")
        else:
            print(f"Failed to read the 5th frame for {filename}")

        # 释放视频对象
        cap.release()

print("Thumbnail generation completed.")