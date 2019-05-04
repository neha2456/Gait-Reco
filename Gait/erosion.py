import cv2
import numpy as np

import os
import glob

# from samba.dcerpc.epmapper.__init__.epm_rhs_osi_clns import epm_rhs_osi_clns
# from skimage.morphology.grey import dilation

train_path = "/home/gaurav/PycharmProjects/Gait/bulky.png"


#
# train_files = sorted(glob.glob(os.path.join(train_path , "*")))
# for filename in (train_files):
#

img = cv2.imread(train_path,0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,  kernel, iterations = 3)

# dilation = cv2.dilate(img,kernel,iterations = 2)
# cv2.imwrite(filename, dilation)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Original',img)
cv2.imshow('Removed Bulky Clothing',erosion)
cv2.waitKey()