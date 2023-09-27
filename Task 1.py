import cv2
import numpy as np

############### Image Processing ###############

#loading image
img = cv2.imread("curfur.jpg")
cv2.imshow('image', img)
print(f"\nimg dimensions = {img.shape} \nimg height = {img.shape[0]} \nimg width = {img.shape[1]} \nimg channels = {img.shape[2]}")
cv2.waitKey(0)

#resizing image
imgResized = cv2.resize(img, (256, 256))
cv2.imshow('imageResized', imgResized)
print(f"\nimgResized dimensions = {imgResized.shape} \nimgResized height = {imgResized.shape[0]} \nimgResized width = {imgResized.shape[1]} \nimgResized channels = {imgResized.shape[2]}")
cv2.waitKey(0)

imgGray = cv2.cvtColor(imgResized, cv2.COLOR_BGR2GRAY)
cv2.imshow("imageGray", imgGray)
print(f"\nimgGray dimensions = {imgGray.shape} \nimgGray height = {imgGray.shape[0]} \nimgGray width = {imgGray.shape[1]}")
# print(f"\nimgGray dimensions = {imgGray.shape} \nimgGray height = {imgGray.shape[0]} \nimgGray width = {imgGray.shape[1]} \nimgGray channels = {imgGray.shape[2]}")
cv2.waitKey(0)

imgBW = cv2.threshold(imgGray, 128, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("imageBW", imgBW)
print(f"\nimgBW dimensions = {imgBW.shape} \nimgBW height = {imgBW.shape[0]} \nimgBW width = {imgBW.shape[1]}")
# print(f"\nimgBW dimensions = {imgBW.shape} \nimgBW height = {imgBW.shape[0]} \nimgBW width = {imgBW.shape[1]} \nimgBW channels = {imgBW.shape[2]}")
cv2.waitKey(0)



