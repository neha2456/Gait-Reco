import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('ff.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,100,0.01,2,useHarrisDetector=True)
corners = np.int0(corners)


count=0
for i in corners:
    count+=1
    x,y = i.ravel()
    cv2.circle(img,(x,y),1,255,-1)


print count
plt.imshow(img),plt.show()