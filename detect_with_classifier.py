from pyimageresearch.detectionhelper import sliding_window,pyramid_generator
from tensorflow.keras.applications import ResNet50, imagenet_utils
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
import numpy as np
import tensorflow as tf
import cv2
import imutils
import time

#Intitialize some constants
Image_path = "images/humming.jpg"
WIDTH = 600
SCALE = 1.5
STEP = 16
WINDOW_SIZE = (200, 150)
INPUT_SIZE = (244, 244)
visualize = False

#load the ResNet50 model with imagenet weights
start = time.time()
model = ResNet50(weights='imagenet', include_top=True)

orig_img = cv2.imread(Image_path)
orig_img = imutils.resize(orig_img, width=WIDTH)
H,W,C = orig_img.shape

pyramid = pyramid_generator(orig_img,SCALE,WINDOW_SIZE)
rois = []
coords = []
for image in pyramid:
    scale_value = W/float(image.shape[1])
    for (x,y,roiOrig) in sliding_window(image,STEP,WINDOW_SIZE):
        x = int(x*scale_value)
        y = int(y*scale_value)
        w = int(WINDOW_SIZE[0]*scale_value)
        h = int(WINDOW_SIZE[1]*scale_value)

        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        rois.append(roi)
        coords.append((x, y, x + w, y + h))
        if visualize:
            clone = orig_img.copy()
            cv2.rectangle(clone,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow("visualize", clone)
            cv2.imshow("ROI", roiOrig)
            cv2.waitKey(100)

rois = np.array(rois, dtype='float32')

preds = model.predict(rois)

preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}
end = time.time()
print(f"Time taken to perform object detection on 1 image : {int(end)-int(start)}")
for (i, p) in enumerate(preds):
    (imagenetId, label, prob) = p[0]
    if prob>=0.95:
        box = coords[i]
        L = labels.get(label,[])
        L.append((box,prob))
        labels[label] = L

for label in labels.keys():
    clone = orig_img.copy()
    clone_2 = orig_img.copy()

    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY),(255,0,0),2)
    cv2.imshow("Before", clone)
    cv2.waitKey(0)
    boxes = np.array([p[0] for p in labels[label]])
    probs = np.array([p[1] for p in labels[label]])
    selected_indices = tf.image.non_max_suppression(boxes, probs, max_output_size=2)
    boxes = tf.gather(boxes, selected_indices)
    #boxes = non_max_suppression(boxes, probs,)
    for x_start, y_start, x_end, y_end in boxes:
        cv2.rectangle(clone_2, (x_start, y_start), (x_end, y_end),
                      (0, 255, 0), 2)
        y = y_start - 10 if y_start - 10 > 10 else y_start + 10
        cv2.putText(clone_2, label, (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2.imshow("After", clone_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()