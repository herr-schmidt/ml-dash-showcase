import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys


img = cv.imread('5cent.png', cv.IMREAD_GRAYSCALE)

img = cv.Canny(img, threshold1=100, threshold2=200)

laplacian = cv.Laplacian(img, cv.CV_64F)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)


magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
angle = np.arctan2(sobely, sobelx) * (180 / np.pi)

# plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 4), plt.imshow(angle, cmap='gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.show()

np.set_printoptions(threshold=sys.maxsize)
print(img)

edge_positions = np.transpose((img > 0).nonzero())
print(edge_positions)

centroid = np.round(np.average(edge_positions, axis=0))

plt.plot(centroid[0], centroid[1], marker='+', color="red")
plt.imshow(img, cmap='gray')
plt.show()