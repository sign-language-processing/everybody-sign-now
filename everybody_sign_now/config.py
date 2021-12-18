OUTPUT_CHANNELS = 3

# Loading 1920x1072 videos
MIN_CROP_SIZE = 900
MAX_CROP_SIZE = 1072

FRAME_DROPOUT = 0.9  # Skip 90% of frames, randomly

BATCH_SIZE = 1  # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment

# Each image is 256x256 in size
RESOLUTION = 256
IMG_HEIGHT = RESOLUTION
IMG_WIDTH = RESOLUTION


LAMBDA = 100
