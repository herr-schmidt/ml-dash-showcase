import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys


img = cv.imread('5cent.png', cv.IMREAD_GRAYSCALE)

img = cv.Canny(img, threshold1=100, threshold2=200)

# laplacian = cv.Laplacian(img, cv.CV_64F)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)


magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
angles = np.round(np.arctan2(sobely, sobelx) * (180 / np.pi))

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

print(angles)

r_table = {theta: [] for theta in np.ndarray.flatten(angles)}

# maybe this can be vectorized somehow
for x in edge_positions:
    r_table[angles[tuple(x)]].append(centroid - x)

print(r_table)

########################## TEST ##########################

test_img = cv.imread('5cent_test.png', cv.IMREAD_GRAYSCALE)
test_img = cv.Canny(test_img, threshold1=100, threshold2=200)

plt.imshow(test_img, cmap='gray')
plt.show()

test_sobelx = cv.Sobel(test_img, cv.CV_64F, 1, 0, ksize=5)
test_sobely = cv.Sobel(test_img, cv.CV_64F, 0, 1, ksize=5)

test_magnitude = np.sqrt(test_sobelx**2.0 + test_sobely**2.0)
test_angles = np.round(np.arctan2(test_sobely, test_sobelx) * (180 / np.pi))

test_edge_positions = np.transpose((test_img > 0).nonzero())

accumulator = np.zeros(test_img.shape)


for x in test_edge_positions:
    if not r_table[test_angles[tuple(x)]]:
        continue

    accumulator_entries = (np.array(r_table[test_angles[tuple(x)]]) + x).astype(int)
    # print(accumulator_entries)
    valid_entries = accumulator_entries[((accumulator_entries[:, 0] >= 0)
                                         & (accumulator_entries[:, 1] >= 0)
                                         & ((accumulator_entries[:, 0] < accumulator.shape[0]))
                                         & ((accumulator_entries[:, 1] < accumulator.shape[1])))]

    accumulator[valid_entries[:, 0], valid_entries[:, 1]] += 1

# plt.plot(centroid[0], centroid[1], marker='+', color="red")
# print(accumulator)
plt.imshow(accumulator, cmap='gray')
plt.show()
