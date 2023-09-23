import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys


img = cv.imread('beer.jpg', cv.IMREAD_GRAYSCALE)

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

edge_positions = np.transpose((img > 0).nonzero())

centroid = np.round(np.average(edge_positions, axis=0))

plt.plot(centroid[1], centroid[0], marker='+', color="red")
plt.imshow(img, cmap='gray')
plt.show()

r_table = {theta: [] for theta in np.ndarray.flatten(angles)}

# maybe this can be vectorized somehow
for x in edge_positions:
    r_table[angles[tuple(x)]].append(centroid - x)

########################## TEST ##########################

test_img = cv.imread('beer_test.jpg', cv.IMREAD_GRAYSCALE)
test_img = cv.Canny(test_img, threshold1=100, threshold2=200)

test_sobelx = cv.Sobel(test_img, cv.CV_64F, 1, 0, ksize=5)
test_sobely = cv.Sobel(test_img, cv.CV_64F, 0, 1, ksize=5)

test_magnitude = np.sqrt(test_sobelx**2.0 + test_sobely**2.0)
test_angles = np.round(np.arctan2(test_sobely, test_sobelx) * (180 / np.pi))

test_edge_positions = np.transpose((test_img > 0).nonzero())


scaling_factors = [1.0] # np.linspace(0.5, 2.0, num=2)
rotation_factors = [0] # [alpha for alpha in range(85, 95)]

accumulator = np.zeros((test_img.shape[0], test_img.shape[1]))

for x in test_edge_positions:
    # if(np.random.uniform(0, 1) > 0.5):
    #     continue
    for (s, s_idx) in zip(scaling_factors, range(0, len(scaling_factors))):
        for alpha in rotation_factors:
            rotated_index = np.mod(test_angles[tuple(x)] - alpha, 360)
            rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

            if not np.round(rotated_index) in r_table:
                continue

            r_vectors = r_table[np.round(rotated_index)]
            if not r_vectors:
                continue

            rotated_and_scaled_r_vectors = np.round(np.transpose(rotation_matrix.dot(np.transpose(r_vectors))) * s)

            accumulator_entries = (rotated_and_scaled_r_vectors + x).astype(int)
            valid_entries = accumulator_entries[((accumulator_entries[:, 0] >= 0)
                                         & (accumulator_entries[:, 1] >= 0)
                                         & ((accumulator_entries[:, 0] < accumulator.shape[0]))
                                         & ((accumulator_entries[:, 1] < accumulator.shape[1])))]

            accumulator[valid_entries[:, 0], valid_entries[:, 1]] += test_magnitude[tuple(x)] + 1


# plt.plot(centroid[0], centroid[1], marker='+', color="red")
# print(accumulator)

max_value = np.max(accumulator)

position = np.transpose((accumulator > max_value - 1).nonzero())

print(position)

# plt.imshow(accumulator, cmap='gray')
plt.imshow(test_img, cmap='gray')
plt.plot(position[0, 1], position[0, 0], marker='+', color="red")
plt.show()
