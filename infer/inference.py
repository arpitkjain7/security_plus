from tensorflow import gather
from tensorflow.image import non_max_suppression
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
import re
from infer.ocr import get_text, get_text_textract
from infer.config import load_config
from db.crud.crud_batch import create_batch

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
        roi = cv2.resize(roi, (244, 244), interpolation=cv2.INTER_CUBIC)
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
    idxs = np.where(proba >= 0.95)
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
    bounding_box_index = -1
    coordinate_list = []
    for ind, percent in pred_dict.items():
        if percent > maximum:
            maximum = percent
            bounding_box_index = ind
            coordinate_list.append(boxes[ind])
    print(f"{bounding_box_index=}")
    if bounding_box_index >= 0:
        coordinates = boxes[bounding_box_index]
        return True, coordinates, coordinate_list
    else:
        return False, (" ", " ", " ", " "), None


def text_postprocess(text):
    text = text.replace("/n", "")
    text = text.replace("/t", "")
    text = re.sub("[^A-Z0-9]", "", text)
    return text


def infer(image_path: str, batch_id: str, frame_id: str = None):
    OutputFolder = os.path.join("output", batch_id)
    os.makedirs(OutputFolder, exist_ok=True)
    model = load_model(model_path)
    image = cv2.imread(image_path)
    # show the output image *after* running NMS
    image = imutils.resize(image, width=500)
    rect = selective_search(image)
    proposals, boxes = get_proposals(image, rect)
    proba = model.predict(proposals)
    object_detected, (startX, startY, endX, endY), coordinate_list = apply_NMS(
        proba, boxes
    )
    print(f"{object_detected=}")
    if object_detected:
        ocr_result_list = []
        db_request = {}
        for num, items in enumerate(coordinate_list):
            clone = image.copy()
            (startX, startY, endX, endY) = items
            y1 = startY - 10 if startY - 10 > 10 else startY + 10
            y2 = endY + 10 if endY + 10 > 10 else endY - 10
            x1 = startX - 10 if startX - 10 > 10 else startX + 10
            x2 = endX + 10 if endX + 10 > 10 else endX - 10
            # cropped = clone[startY:endY, startX:endX]
            cropped = clone[y1:y2, x1:x2]
            # cropped = cv2.medianBlur(cropped, 5)
            # ret, cropped = cv2.threshold(cropped, 150, 255, cv2.THRESH_BINARY)
            # gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            # gray = cv2.bitwise_not(gray)
            # # threshold the image, setting all foreground pixels to
            # # 255 and all background pixels to 0
            # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            out_path = os.path.join(OutputFolder, f"cropped_{num}.jpg")
            cv2.imwrite(out_path, cropped)
            ocr_result = get_text(out_path)
            # ocr_result = get_text_textract(out_path)
            # print(ocr_result)
            ocr_result = text_postprocess(ocr_result)
            # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # text = f"{ocr_result}"
            # cv2.putText(
            #     image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1
            # )
            # cv2.imwrite(os.path.join(OutputFolder, "final.jpg"), image)
            # cv2.imshow("output", image)
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()
            frame_id = batch_id if frame_id is None else frame_id
            db_request.update(
                {
                    "batch_id": batch_id,
                    "frame_id": frame_id,
                    "licence_plate_path": out_path,
                    "licence_num": ocr_result,
                }
            )
            ocr_result_list.append(ocr_result)
        print(f"length of ocr result list : {len(ocr_result_list)}")
        print(f"{ocr_result_list=}")
        print(f"length of ocr result list item : {len(ocr_result_list[0])}")
        if len(ocr_result_list[0]) > 4:
            create_batch(db_request)
        return db_request
    else:
        return "No vehicle detected"