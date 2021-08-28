OUTPUT_CHANNELS = 3

RESOLUTION = 256

BUFFER_SIZE = 5000
BATCH_SIZE = 1  # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
# Each image is 256x256 in size
IMG_HEIGHT = RESOLUTION
IMG_WIDTH = RESOLUTION

LAMBDA = 100