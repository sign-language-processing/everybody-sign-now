import cv2
import numpy as np
from pose_format import Pose
from pose_format.pose_header import PoseHeaderDimensions
from pose_format.pose_visualizer import PoseVisualizer

from everybody_sign_now.config import RESOLUTION
from everybody_sign_now.data import normalize_img

import tensorflow as tf
from tqdm import tqdm

from everybody_sign_now.model import discriminator, discriminator_optimizer, generator,   generator_optimizer


def load_model():
    checkpoint_dir = 'training_checkpoints'
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    print("latest_checkpoint", latest_checkpoint)
    checkpoint.restore(latest_checkpoint)

    return generator


def unnormalize_img(img):
    ranged = (img.numpy() * 0.5) + 0.5  # Normalization to range [0, 1]
    ranged *= 255
    print(ranged[0])
    return ranged.astype(np.uint8)


if __name__ == "__main__":
    with open("example.pose", "rb") as f:
        pose = Pose.read(f.read())

    pose.body.data = pose.body.data / (max(pose.header.dimensions.height, pose.header.dimensions.width) / RESOLUTION)
    pose.header.dimensions = PoseHeaderDimensions(RESOLUTION, RESOLUTION)

    visualizer = PoseVisualizer(pose, thickness=1)
    frames = list(visualizer.draw(background_color=(200, 200, 255)))
    batch = normalize_img(np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]))
    print("batch shape", batch.shape)

    model = load_model()
    print("Predicting")
    prediction_frames = unnormalize_img(model(batch, training=True))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('prediction.avi', fourcc, pose.body.fps, (RESOLUTION, RESOLUTION))

    for frame in tqdm(prediction_frames):
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
