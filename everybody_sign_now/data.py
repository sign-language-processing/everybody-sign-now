import itertools
import math
import os
import random
from typing import List

import cv2
import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions
from pose_format.pose_visualizer import PoseVisualizer

from tqdm import tqdm

from everybody_sign_now.config import FRAME_DROPOUT, MIN_CROP_SIZE, MAX_CROP_SIZE, RESOLUTION


def load_video(video_path: str):
    # Read a video as iterable
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


# Normalizing the images to [-1, 1]
def normalize_img(img):
    import tensorflow as tf

    return (tf.cast(img, tf.float32) / 127.5) - 1


def video_dataset(video_path: str, pose_paths: List[str], background_color=(255, 255, 255)):
    """
    Iteratively reads a video file.
    Performs frame dropout - ignores FRAME_DROPOUT% of the frames
    Crops the frame between MIN_CROP_SIZE and MAX_CROP_SIZE and resizes it to be (RESOLUTION, RESOLUTION)
    Crops the pose accordingly, and resizes it accordingly.
    Given 50% chance, out of frame landmarks are removed.
    Generates the pose image
    Given 50% chance, flips the resulting images horizontally.
    """

    # Load poses
    poses = []
    for pose_path in pose_paths:
        with open(pose_path, "rb") as f:
            pose = Pose.read(f.read())
            pose.header.dimensions = PoseHeaderDimensions(RESOLUTION, RESOLUTION)

            poses.append(pose)

    while True:
        # Load video
        frames = load_video(video_path)

        for i, frame in enumerate(frames):
            if random.random() < FRAME_DROPOUT:
                continue

            # Set frame crop
            crop_size = random.randint(MIN_CROP_SIZE, MAX_CROP_SIZE)
            h, w, _ = frame.shape
            crop_start_h = random.randint(0, h - crop_size)
            crop_start_w = random.randint(0, w - crop_size)
            cropped_frame = frame[crop_start_h:crop_start_h + crop_size, crop_start_w:crop_start_w + crop_size]
            cropped_frame = cv2.resize(cropped_frame, (RESOLUTION, RESOLUTION))

            # "crop" pose
            pose = random.choice(poses)
            data = pose.body.data.data[i:i + 1, :, :, :2]
            data = (data - np.array([crop_start_w, crop_start_h])) / (crop_size / RESOLUTION)

            conf = pose.body.confidence[i:i + 1, :, :]

            # Remove out-of-frame keypoints
            if random.random() > 0.5:
                data = np.where(data < RESOLUTION, data, 0)

                conf_map = data.min(axis=-1, initial=math.inf)
                conf = np.where(conf_map > 0, conf, 0)

            # Skip empty frames
            if conf.sum() == 0:
                continue

            new_body = NumPyPoseBody(fps=1, data=data, confidence=conf)
            pose = Pose(pose.header, new_body)

            visualizer = PoseVisualizer(pose, thickness=1)
            cropped_pose_frame = cv2.cvtColor(next(iter(visualizer.draw(background_color=background_color))),
                                              cv2.COLOR_BGR2RGB)

            # Left right flip, 50%
            if random.random() > 0.5:
                cropped_frame = cv2.flip(cropped_frame, 1)
                cropped_pose_frame = cv2.flip(cropped_pose_frame, 1)

            yield cropped_pose_frame, cropped_frame


def multi_video_dataset(video_datasets):
    while True:
        for d in video_datasets:
            yield next(d)


def project_dataset(colors: dict = {}):
    # base_dir = Path(__file__).parent.parent.joinpath('data') # TODO
    base_dir = "/home/nlp/amit/WWW/datasets/GreenScreen/mp4/"

    video_datasets = []
    for name in os.listdir(base_dir):
        name_files = os.listdir(os.path.join(base_dir, name))
        for video in [f for f in name_files if f.endswith(".mp4")]:
            video_path = os.path.join(base_dir, name, video)
            [video_name, _] = video.split(".")
            pose_paths = [os.path.join(base_dir, name, f) for f in name_files
                          if f.startswith(video_name) and f.endswith(".pose")]

            if len(pose_paths) > 0:
                color = colors[name] if name in colors else (255, 255, 255)
                print("Dataset:", video_path, pose_paths, color)
                video_datasets.append(video_dataset(video_path, pose_paths, color))

    return multi_video_dataset(video_datasets)


def test_dataset():
    base_dir = "/home/nlp/amit/WWW/datasets/GreenScreen/mp4/Maayan_1/"
    video_path = os.path.join(base_dir, "CAM3.mp4")
    pose_paths = [os.path.join(base_dir, "CAM3.openpose.pose")]
    return video_dataset(video_path, pose_paths, (200, 200, 255))


def batch_dataset(dataset, batch_size):
    while True:
        batch = [next(dataset) for _ in range(batch_size)]
        b0 = normalize_img(np.stack([b[0] for b in batch]))
        b1 = normalize_img(np.stack([b[1] for b in batch]))
        yield b0, b1


if __name__ == "__main__":
    dataset = project_dataset({"Amit": (255, 200, 200), "Maayan_1": (200, 200, 255), "Maayan_2": (200, 200, 255)})

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.avi', fourcc, 10, (RESOLUTION * 2, RESOLUTION))

    for pose_frame, video_frame in tqdm(itertools.islice(dataset, 0, 1000)):
        video.write(cv2.cvtColor(np.hstack((pose_frame, video_frame)), cv2.COLOR_RGB2BGR))

    video.release()

    # base_dir = "/home/nlp/amit/WWW/datasets/GreenScreen/mp4/Maayan_1/"
    # dataset = video_dataset(base_dir + "CAM3.mp4", [base_dir + "CAM3.holistic.pose"], (255, 200, 200))
    #
    # pose_frame, video_frame = next(dataset)
    # cv2.imwrite("test_pose_frame.png", cv2.cvtColor(pose_frame, cv2.COLOR_RGB2BGR))
    # cv2.imwrite("test_video_frame.png", cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))
