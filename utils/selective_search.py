import cv2
import time
import random
image_path = "images/humming.jpg"
mode = 'quality'
img = cv2.imread(image_path)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)

if mode == 'fast':
    ss.switchToSelectiveSearchFast()
elif mode == 'quality':
    ss.switchToSelectiveSearchQuality()
start = time.time()
rectangles = ss.process()
end = time.time()

print(f"time taken : {int(end)-int(start)} seconds")
print(f"total regions proposed : {len(rectangles)}")

for i in range(0, len(rectangles), 100):
    clone = img.copy()
    for x,y,w,h in rectangles[i:i+100]:
        color = [random.randint(0,255) for f in range(0,3)]
        cv2.rectangle(clone,(x,y),(x+w,y+h),color,2)
    cv2.imshow('selective search',clone)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break
