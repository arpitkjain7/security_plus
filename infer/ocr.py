import cv2
import os
import pytesseract
from infer.ocr_textract import detect_text


def get_text_testing(path: str):
    core_dir = "OCR"
    for images in os.listdir(core_dir):
        if images != ".DS_Store":
            print(images)
            img_cv = cv2.imread(os.path.join(core_dir, images))

            # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
            # we need to convert from BGR to RGB format/mode:
            img_grey = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("Gray image", img_grey)
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()
            print(pytesseract.image_to_string(img_grey))


def get_text(path: str):

    # deskewed_page = preprocess_image(file_path=os.path.join(path, images))
    img_cv = cv2.imread(path)
    # deskewed_page = preprocess_image_file(img_cv)

    # # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
    # # we need to convert from BGR to RGB format/mode:
    img_grey = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray image", deskewed_page)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()
    custom_config = r"--oem 3 --psm 6"
    ocr_result = pytesseract.image_to_string(img_grey, config=custom_config)
    # ocr_result = pytesseract.image_to_string(img_grey)
    # ocr_result = detect_text(os.path.join(path, images))
    return ocr_result
