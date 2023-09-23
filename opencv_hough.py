import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys

img = cv.imread('beer_bottle.jpg', cv.IMREAD_GRAYSCALE)

edge_image = cv.Canny(img, threshold1=100, threshold2=200)

plt.imshow(edge_image, cmap='gray')
plt.show()

ght = cv.createGeneralizedHoughBallard()
ght.setTemplate(img)

ght.setMinDist(500)
ght.setLevels(360)
ght.setMaxBufferSize(8192)
ght.setVotesThreshold(30)
ght.setCannyLowThresh(100)
ght.setCannyHighThresh(200)



# test_img = cv.imread('beer_test.jpg', cv.IMREAD_GRAYSCALE)
# test_img = cv.Canny(test_img, threshold1=100, threshold2=200)

# while True:

# positions = ght.detect(test_img)[0][0]

# plt.imshow(test_img, cmap='gray')
# plt.plot(positions[0][0], positions[0][1], marker='+', color="red")
# plt.show()

# cap = cv.VideoCapture('beer_video.mp4')
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 768)
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    detection = ght.detect(image)
    print(detection)

    gray = cv.cvtColor(frame, cv.COLOR_RGB2RGBA)

    if detection[0] is not None:
        print(detection[0][0][0][0])
        cv.circle(img=gray, center=(int(detection[0][0][0][0]), int(detection[0][0][0][1])), radius=150, color=(0, 0, 255), thickness=20)
    
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break


cap.release()
cv.destroyAllWindows()