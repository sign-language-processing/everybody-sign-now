import os

import cv2
from tqdm import tqdm

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('progress.avi', fourcc, 5, (1500, 500))

progress = sorted(os.listdir("progress"), key=lambda f: int(f.split(".")[0]))
print(progress)

for frame in tqdm(progress):
    video.write(cv2.imread("progress/" + frame))

video.release()
