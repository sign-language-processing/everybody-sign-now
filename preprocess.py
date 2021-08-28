import os

import cv2
import numpy as np
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from tqdm import tqdm

from config import RESOLUTION

os.makedirs("frames", exist_ok=True)

POSE = "openpose"

# Load pose
buffer = open("data/" + POSE + ".pose", "rb").read()
pose = Pose.read(buffer, NumPyPoseBody)
assert pose.header.dimensions.width == pose.header.dimensions.height

# Rescale pose
ratio = RESOLUTION / pose.header.dimensions.width
pose.body.data = pose.body.data * ratio
pose.header.dimensions.width = pose.header.dimensions.height = RESOLUTION

visualizer = PoseVisualizer(pose)

# Load video
vid_capture = cv2.VideoCapture('data/video.mp4')

for i, pose_frame in tqdm(enumerate(visualizer.draw(background_color=(255, 255, 255)))):
    _, video_frame = vid_capture.read()
    video_frame = cv2.resize(video_frame, (RESOLUTION, RESOLUTION))

    joined = np.concatenate([video_frame, pose_frame], axis=1)
    cv2.imwrite("frames/" + str(i) + ".png", joined)

# Release the video capture object
vid_capture.release()
