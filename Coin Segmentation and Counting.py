import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# loading the image
img = cv2.imread('coins.png', 1)

# segmentation
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 160, 255, cv2.THRESH_BINARY)

# Get the coordinates of pixels with intensity value 0
non_zero_coords = np.argwhere(binary_image == 0)

# Prepare the data
result_array = np.hstack((non_zero_coords, np.zeros((non_zero_coords.shape[0], 1))))

# Counting
model = DBSCAN(eps=1.5, min_samples=6).fit(result_array)
labels = np.array(model.labels_)
num_clusters = np.unique(labels).shape[0] - 1



#writing the number of coins
position = (50, 50)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 0, 0)  # (B, G, R) color
line_type = 2
cv2.putText(binary_image, f"There are {num_clusters} coins", position, font, font_scale, font_color, line_type)

cv2.imshow('Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()