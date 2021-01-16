def compute_iou(boxA, boxB):
    # boxA = (startx, starty, endx, endy)
    # Calculate the intersection coordinates between BoxA and BoxB
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # Calculate Intersection area between BoxA and BoxB
    intersectionArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaOfBoxA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaOfBoxB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # Calculate the IOU
    iou = intersectionArea / float(areaOfBoxA + areaOfBoxB - intersectionArea)
    return iou


# iou = compute_iou([39, 63, 50, 112], [54, 66, 198, 114])
# print(iou)
