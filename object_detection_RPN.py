from tensorflow.keras.applications import ResNet50, imagenet_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import time

image = cv2.imread("images/humming.jpg")
labelfilters = ["hummingbird", "dog"]


def selective_search(image, mode="fast"):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    elif mode == "quality":
        ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    return rects


start = time.time()
model = ResNet50(weights="imagenet")

H, W, C = image.shape
proposals = []
boxes = []

rects = selective_search(image, "fast")

for x, y, w, h in rects:
    if w / W < 0.1 or h / H < 0.1:
        continue
    roi = image[y : y + h, x : x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (244, 244))
    roi = img_to_array(roi)
    roi = preprocess_input(roi)
    proposals.append(roi)
    boxes.append((x, y, w, h))
proposals = np.array(proposals)
preds = model.predict(proposals)
preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}
end = time.time()
print(f"time taken for predition : {int(end)-int(start)} seconds")
for i, p in enumerate(preds):
    imageId, label, prob = p[0]
    if prob > 0.95:
        x, y, w, h = boxes[i]
        box = (x, y, x + w, y + h)
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

for label in labels.keys():
    clone = image.copy()
    for (box, prob) in labels[label]:
        startX, startY, endX, endY = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (255, 0, 0), 2)
    # cv2.imshow("before", clone)
    # cv2.waitKey(0)

    clone_2 = image.copy()
    boxes = np.array([p[0] for p in labels[label]])
    probs = np.array([p[1] for p in labels[label]])
    segments = tf.image.non_max_suppression(boxes, probs, max_output_size=1)
    boxes = tf.gather(boxes, segments)
    for x_start, y_start, x_end, y_end in boxes:
        cv2.rectangle(clone_2, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        y = y_start - 10 if y_start - 10 > 10 else y_start + 10
        cv2.putText(
            clone_2, label, (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2
        )
    cv2.imshow("After", clone_2)
    cv2.waitKey(0)
