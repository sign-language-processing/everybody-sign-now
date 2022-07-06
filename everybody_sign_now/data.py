import itertools
import math
import os
import random
from typing import List
from itertools import islice

import cv2
import numpy as np
import numpy.ma as ma
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions, PoseHeader
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


def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def shoulders_indexes(pose_header: PoseHeader):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return (pose_header._get_point_index("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                pose_header._get_point_index("POSE_LANDMARKS", "LEFT_SHOULDER"))

    if pose_header.components[0].name == "BODY_135":
        return (pose_header._get_point_index("BODY_135", "RShoulder"),
                pose_header._get_point_index("BODY_135", "LShoulder"))

    if pose_header.components[0].name == "pose_keypoints_2d":
        return (pose_header._get_point_index("pose_keypoints_2d", "RShoulder"),
                pose_header._get_point_index("pose_keypoints_2d", "LShoulder"))


def video_dataset(video_path: str, pose_paths: List[str], background_color=(255, 255, 255), aspect_ratio_std=0.05):
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

        for batch in batcher(enumerate(frames), batch_size=1):  # For sequential prediction
            if random.random() < FRAME_DROPOUT:
                continue

            indexes, frames = list(zip(*batch))
            frames = np.array(frames)

            _, h, w, _ = frames.shape

            # Get pose data
            pose = random.choice(poses)
            data = pose.body.data.data[indexes, :, :, :2]
            conf = pose.body.confidence[indexes, :, :]
            # Skip empty frames
            if conf.sum() == 0:
                continue

            r_shoulder_i, l_shoulder_i = shoulders_indexes(pose.header)
            r_shoulder = data[:, 0, r_shoulder_i]
            l_shoulder = data[:, 0, l_shoulder_i]
            r_shoulder_x = int(ma.mean(r_shoulder[:, 0], axis=0))
            l_shoulder_x = int(ma.mean(l_shoulder[:, 0], axis=0))
            shoulders_x = abs(int((r_shoulder_x + l_shoulder_x) / 2))
            shoulders_y = abs(int(ma.mean((l_shoulder[:, 1] + r_shoulder[:, 1]) / 2, axis=0)))
            shoulder_width = abs(r_shoulder_x - l_shoulder_x)
            offset = shoulder_width
            crop_start_w = crop_start_h = crop_size_w = crop_size_h = -1  # init params
            # Make sure crom is not out of frame
            while crop_start_w < 0 \
                    or crop_start_w + crop_size_w > w \
                    or crop_start_h < 0 \
                    or crop_start_h + crop_size_h > h:
                if offset < shoulder_width * 0.3:
                    offset = 0
                    break

                crop_size_w = int(3 * offset)
                crop_size_h = int(crop_size_w * float(np.random.normal(1, aspect_ratio_std)))
                crop_start_w = int(shoulders_x - crop_size_w / 2)
                crop_start_h = max(0, int(shoulders_y - crop_size_h / 2))
                offset *= 0.95

            if offset == 0:
                continue

            # # Set frame crop
            # crop_size = random.randint(MIN_CROP_SIZE, MAX_CROP_SIZE)
            # crop_start_h = random.randint(0, h - crop_size)
            # crop_start_w = random.randint(0, w - crop_size)

            # Crop frames
            cropped_frames = frames[:, crop_start_h:crop_start_h + crop_size_h, crop_start_w:crop_start_w + crop_size_w]

            # print("shape", cropped_frames.shape)

            # Resize frames
            cropped_frames = np.array([cv2.resize(frame, (RESOLUTION, RESOLUTION))
                                       for frame in cropped_frames])

            # "crop" pose
            data = (data - np.array([crop_start_w, crop_start_h])) / (np.array([crop_size_w, crop_size_h]) / RESOLUTION)

            # # Remove out-of-frame keypoints 10%
            # if random.random() > 0.9:
            #     data = np.where(data < RESOLUTION, data, 0)
            #
            #     conf_map = data.min(axis=-1, initial=math.inf)
            #     conf = np.where(conf_map > 0, conf, 0)

            new_body = NumPyPoseBody(fps=1, data=data, confidence=conf)
            pose = Pose(pose.header, new_body)

            visualizer = PoseVisualizer(pose, thickness=1)
            cropped_pose_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in
                                   visualizer.draw(background_color=background_color)]
            cropped_pose_frames = np.array(cropped_pose_frames)

            # # Left right flip, 50%. Good for pretraining, but flickers if left in final model
            # if random.random() > 0.5:
            #     cropped_frames = np.array([cv2.flip(frame, 1) for frame in cropped_frames])
            #     cropped_pose_frames = np.array([cv2.flip(frame, 1) for frame in cropped_pose_frames])

            yield cropped_pose_frames[0], cropped_frames[0]  # only 1 frame


def multi_video_dataset(video_datasets):
    while True:
        for d in video_datasets:
            yield next(d)


def project_dataset(colors: dict = {}):
    # base_dir = Path(__file__).parent.parent.joinpath('data') # TODO
    base_dir = "/home/nlp/amit/WWW/datasets/GreenScreen/mp4/"

    video_ext = "_norm.mp4"
    video_datasets = []
    for name in os.listdir(base_dir):
        name_files = os.listdir(os.path.join(base_dir, name))
        for video in [f for f in name_files if f.endswith(video_ext)]:
            video_path = os.path.join(base_dir, name, video)
            video_name = video[:-len(video_ext)]
            pose_paths = [os.path.join(base_dir, name, f) for f in name_files
                          if f.startswith(video_name) and f.endswith(".pose")]

            if len(pose_paths) > 0:
                color = colors[name] if name in colors else (255, 255, 255)
                print("Dataset:", video_path, pose_paths, color)
                video_datasets.append(video_dataset(video_path, pose_paths, color))

    return multi_video_dataset(video_datasets)


def test_dataset():
    base_dir = "/home/nlp/amit/WWW/datasets/GreenScreen/mp4/Maayan_1/"
    video_path = os.path.join(base_dir, "CAM3_norm.mp4")
    pose_paths = [os.path.join(base_dir, "CAM3.openpose.pose")]
    return video_dataset(video_path, pose_paths, (200, 200, 255))


def batch_dataset(dataset, batch_size):
    while True:
        batch = [next(dataset) for _ in range(batch_size)]
        b0 = normalize_img(np.stack([b[0] for b in batch]))
        b1 = normalize_img(np.stack([b[1] for b in batch]))
        yield b0, b1


if __name__ == "__main__":
    # dataset = project_dataset({"Amit": (255, 200, 200), "Maayan_1": (200, 200, 255), "Maayan_2": (200, 200, 255)})
    dataset = test_dataset()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.avi', fourcc, 10, (RESOLUTION * 2, RESOLUTION))

    for pose_frame, video_frame in tqdm(itertools.islice(dataset, 0, 100)):
        video.write(cv2.cvtColor(np.hstack((pose_frame, video_frame)), cv2.COLOR_RGB2BGR))

    video.release()

    # base_dir = "/home/nlp/amit/WWW/datasets/GreenScreen/mp4/Maayan_1/"
    # dataset = video_dataset(base_dir + "CAM3.mp4", [base_dir + "CAM3.holistic.pose"], (255, 200, 200))
    #
    # pose_frame, video_frame = next(dataset)
    # cv2.imwrite("test_pose_frame.png", cv2.cvtColor(pose_frame, cv2.COLOR_RGB2BGR))
    # cv2.imwrite("test_video_frame.png", cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))
