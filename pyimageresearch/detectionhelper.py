import imutils
import cv2
def sliding_window(image, step, ws):
    for y in range(0,image.shape[0]-ws[1],step):
        for x in range(0, image.shape[1]-ws[0], step):
            yield (x,y,image[y : y + ws[1],x : x + ws[0]])

def pyramid_generator(image, scale=1.5, minSize=(244,244)):
    yield image
    while True:
        w = int(image.shape[1]/scale)
        image = imutils.resize(image, width=w)
        if image.shape[0]<minSize[0] or image.shape[1]<minSize[1]:
            break
        yield image