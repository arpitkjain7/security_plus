from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import time
import os
from datetime import datetime
from infer.inference import infer


def get_video_frame(video_path: str, batch_id: str):
    # video = "video/parking.mp4"
    stream = cv2.VideoCapture(video_path)
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
        frame = imutils.resize(frame, width=2000)
        # show the frame and update the FPS counter
        if frame_num % 100 == 0:
            frame_id = f"{batch_id}_{frame_num}"
            frame_path = f"{folder}/{frame_num}.jpg"
            cv2.imwrite(frame_path, frame)
            result = infer(image_path=frame_path, batch_id=frame_id)
        # cv2.imshow("Frame", frame)
        # cv2.waitKey(1)
        frame_num += 1
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    stream.release()
    # cv2.destroyAllWindows()
    return {"Status": "Success", "Batch_id": batch_id}
