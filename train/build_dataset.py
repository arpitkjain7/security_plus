from iou import compute_iou
from bs4 import BeautifulSoup
import cv2
import os

totalPositive = 0
totalNegative = 0
for image in os.listdir("car/images"):
    print(f"processing image : {image}")
    file_name = image.split(".")[0]
    image_path = os.path.join("car/images", image)
    print(image_path)
    img = cv2.imread(image_path)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    proposed_rects = []

    for (x, y, w, h) in rects:
        proposed_rects.append((x, y, x + w, y + h))

    annotation_path = os.path.join("car/annotations", file_name + ".xml")
    f = open(annotation_path, mode="r")
    soup = BeautifulSoup(f, "html.parser")
    w = int(soup.find("width").string)
    h = int(soup.find("height").string)
    gtBoxes = []
    for o in soup.find_all("object"):
        x_min = int(o.find("xmin").string)
        y_min = int(o.find("ymin").string)
        x_max = int(o.find("xmax").string)
        y_max = int(o.find("ymax").string)
        label = o.find("name").string

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(x_max, w)
        y_max = min(y_max, h)

        gtBoxes.append((x_min, y_min, x_max, y_max))
    positiveROI = 0
    negativeROI = 0
    for proposed_rects in proposed_rects[:2000]:
        propStartX, propStartY, propEndX, propEndY = proposed_rects
        roi = None
        OutputPath = None
        for gtBox in gtBoxes:
            gtStartX, gtStartY, gtEndX, gtEndY = gtBox
            iou = compute_iou(gtBox, proposed_rects)
            """
            Only take 30 positive ROI which have IOU greater than 70%
            i.e. take only those sections of image which overlap more than 70%
            of the area with the annotation
            """
            if iou >= 0.70 and positiveROI < 30:
                roi = img[propStartY:propEndY, propStartX:propEndX]
                OutputPath = os.path.join("dataset/licence", f"{totalPositive}.jpg")
                totalPositive += 1
                positiveROI += 1
            """
            If coordinates of annotated section is smaller than the propsed roi
            this calls for complete overlap
            """
            fullOverlap = (
                propStartX >= gtStartX
                and propStartY >= gtStartY
                and propEndX <= gtEndX
                and propEndY <= gtEndY
            )
            """
            Only take 10 negative ROI which have IOU less than 5%
            i.e. take only those sections of image which overlap less than 5%
            of the area with the annotation
            """
            if not fullOverlap and iou <= 0.05 and negativeROI < 10:
                roi = img[propStartY:propEndY, propStartX:propEndX]
                OutputPath = os.path.join("dataset/no_licence", f"{totalNegative}.jpg")
                totalNegative += 1
                negativeROI += 1
            if roi is not None and OutputPath is not None:
                roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(OutputPath, roi)
