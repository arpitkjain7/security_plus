import tensorflow as tf
from tensorflow.image import non_max_suppression
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
from datetime import datetime

start_time = datetime.now()
model = load_model("model/object_detection.h5")
lb = pickle.loads(open("label_encoder/label_encoder.pickle", "rb").read())

image = cv2.imread("raccoons/images/raccoon-1.jpg")
image = imutils.resize(image, width=500)
print(image.shape)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rect = ss.process()

proposals = []
boxes = []
for x, y, w, h in rect[:200]:
    roi = image[y : y + h, x : x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_CUBIC)
    roi = img_to_array(roi)
    # roi = preprocess_input(roi)
    proposals.append(roi)
    boxes.append((x, y, x + w, y + h))

proposals = np.array(proposals)
boxes = np.array(boxes)
print(proposals.shape)
proba = model.predict(proposals)

# find the index of all predictions that are positive for the
# "raccoon" class
print("[INFO] applying NMS...")
labels = lb.classes_[np.argmax(proba, axis=1)]
idxs = np.where(labels == "raccoon")[0]
# use the indexes to extract all bounding boxes and associated class
# label probabilities associated with the "raccoon" class
boxes = boxes[idxs]
proba = proba[idxs][:, 1]
# further filter indexes by enforcing a minimum prediction
# probability be met
idxs = np.where(proba >= 0.75)
boxes = boxes[idxs]
proba = proba[idxs]

# clone the original image so that we can draw on it
clone = image.copy()
# loop over the bounding boxes and associated probabilities
for (box, prob) in zip(boxes, proba):
    # draw the bounding box, label, and probability on the image
    (startX, startY, endX, endY) = box
    cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text = "Raccoon: {:.2f}%".format(prob * 100)
    cv2.putText(
        clone, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2
    )
# show the output after *before* running NMS
# cv2.imshow("Before NMS", clone)
# cv2.waitKey(100)
# cv2.destroyAllWindows()

# run non-maxima suppression on the bounding boxes
boxIdxs = non_max_suppression(boxes, proba, max_output_size=1)
boxes = tf.gather(boxes, boxIdxs)
# loop over the bounding box indexes
for i in boxIdxs:
    # draw the bounding box, label, and probability on the image
    (startX, startY, endX, endY) = boxes[i]
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text = "Raccoon: {:.2f}%".format(proba[i] * 100)
    cv2.putText(
        image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2
    )
# show the output image *after* running NMS
end_time = datetime.now()
time_take = (end_time - start_time).seconds
print(f"bounding box:{float(startX),float(startY),float(endX),float(endY)}")
print(f"time take to infer : {time_take} seconds")
