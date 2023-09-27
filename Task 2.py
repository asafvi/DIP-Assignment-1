import cv2 as cv
import numpy as np

img = cv.imread('coins.jpg')
cv.imshow("Original", img)

resized_img = cv.resize(img, (256, 256))

# Convertin to grayscale
gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

# Applying Gaussian Blur
blur = cv.GaussianBlur(gray, (21, 21), 0)
cv.imshow("Blurred", blur)

# Thresholding
x, thresh = cv.threshold(blur, 125, 255, cv.THRESH_BINARY)
cv.imshow("Thresh", thresh)

# Detecting Edges
edges = cv.Canny(thresh, 30, 150)

contours, x = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

min_area = 100
filtered_contours = [contour for contour in contours if cv.contourArea(contour) > min_area]

# Draw the filtered contours on a copy of the original image
image_with_contours = resized_img.copy()
cv.drawContours(image_with_contours, filtered_contours, -1, (0, 255, 0), 1)

# Count the number of coins
num_coins = len(filtered_contours)
print(num_coins)

# Reduce font size
font_scale = 0.5

# Reduce font thickness
font_thickness = 1

cv.putText(image_with_contours, f'Number of Coins: {num_coins}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
cv.imshow('Coins with Contours', image_with_contours)
cv.waitKey(0)
cv.destroyAllWindows()

cv.waitKey(0)