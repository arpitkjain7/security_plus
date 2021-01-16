from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import time
import os
from datetime import datetime

video = "video/22.mp4"
stream = cv2.VideoCapture(video)
fps = FPS().start()
batch_id = str(int(datetime.now().timestamp() * 1000))
folder = f"frames/{batch_id}"
os.makedirs(folder, exist_ok=True)
# loop over frames from the video file stream
frame_num = 0
while True:
    # grab the frame from the threaded video file stream
    (grabbed, frame) = stream.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    # resize the frame and convert it to grayscale (while still
    # retaining 3 channels)
    frame = imutils.resize(frame, width=1000)
    # show the frame and update the FPS counter
    if frame_num % 75 == 0:
        cv2.imwrite(f"{folder}/{frame_num}.jpg", frame)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    frame_num += 1
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()
