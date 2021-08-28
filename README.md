# Everybody Sign Now

This repository aims to train an in-browser real-time image translation model from pose estimation to videos.

The code is a port of [Tensorflow's pix2pix tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix).

## Data

High resolution green screen videos will soon be available.

Mandatory files:
- `data/video.mp4`
- `data/openpose.pose`

## Preprocessing

Run `python preprocess.py`.

This creates a directory "`frames`" with side-by-side images for the pose and video.

## Training

Run `python train.py`

This will train for a long while, and log each epoch result in a "`progress`" directory.
Once satisfied with the result, the script can be killed.

![Progress Sample](progress_sample.png)

## Converting to `tfjs`

Run `./convert_to_tfjs.sh`

This will create a `web_model` directory with the model in tfjs, quantized to `float16`.