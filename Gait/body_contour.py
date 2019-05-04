import numpy as np
import cv2

im = cv2.imread('remove.png')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
_,contours, hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(im, contours, -1, (0,255,0), 2)
cv2.imshow("Body Outline", im)
cv2.waitKey()