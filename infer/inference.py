import tensorflow as tf
from tensorflow import gather
from tensorflow.image import non_max_suppression
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
from infer.ocr import get_text
from infer.config import load_config

config = load_config()
model_path = config.get("model_path")
label_encoder_path = config.get("label_encoder_path")


def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rect = ss.process()
    return rect


def get_proposals(image, rect):
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
    return proposals, boxes


def apply_NMS(proba, boxes):
    print("[INFO] applying NMS...")
    lb = pickle.loads(open(label_encoder_path, "rb").read())
    labels = lb.classes_[np.argmax(proba, axis=1)]
    idxs = np.where(labels == "licence")[0]
    # use the indexes to extract all bounding boxes and associated class
    # label probabilities associated with the "raccoon" class
    boxes = boxes[idxs]
    proba = proba[idxs][:, 0]
    # further filter indexes by enforcing a minimum prediction
    # probability be met
    idxs = np.where(proba >= 0.8)
    boxes = boxes[idxs]
    proba = proba[idxs]

    # loop over the bounding boxes and associated probabilities
    for (box, prob) in zip(boxes, proba):
        (startX, startY, endX, endY) = box
    boxIdxs = non_max_suppression(boxes, proba, max_output_size=5)
    # print(boxIdxs)
    boxes = gather(boxes, boxIdxs)
    n = 0
    pred_dict = {}
    for _, i in zip(boxes, boxIdxs):
        pred_dict.update({n: proba[i]})
        # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # y = startY - 10 if startY - 10 > 10 else startY + 10
        # text = "Licence:"
        # cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        n += 1
    maximum = 0
    bounding_box_index = 0
    for ind, percent in pred_dict.items():
        if percent > maximum:
            maximum = percent
            bounding_box_index = ind
    coordinates = boxes[bounding_box_index]
    return coordinates


def infer(image_path: str, batch_id: str):
    OutputFolder = os.path.join("output", batch_id)
    os.makedirs(OutputFolder, exist_ok=True)
    model = load_model(model_path)
    image = cv2.imread(image_path)
    # show the output image *after* running NMS
    image = imutils.resize(image, width=500)
    rect = selective_search(image)
    proposals, boxes = get_proposals(image, rect)
    proba = model.predict(proposals)
    (startX, startY, endX, endY) = apply_NMS(proba, boxes)

    cropped = image[startY:endY, startX:endX]
    cv2.imwrite(os.path.join(OutputFolder, "cropped.jpg"), cropped)
    ocr_result = get_text(OutputFolder)
    print(ocr_result)
    return ocr_result
    # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # y = startY - 10 if startY - 10 > 10 else startY + 10
    # text = f"{ocr_result}"
    # cv2.putText(
    #     image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2
    # )


# infer("/Users/arpitkjain/Desktop/Data/POC/security_plus/test-data/abhilash-1.jpeg")
