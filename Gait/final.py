import cv2
import numpy as np
import glob
import os

import numpy as np


import matplotlib.pyplot as plt


from matplotlib import style


fig = plt.figure()
ax1 = fig.add_subplot(111)
from sklearn.svm import SVC
clf = SVC( probability=True,
          tol=1e-3)

def cluster(coord,label):

    style.use("ggplot")


    X=np.asarray(coord)
    y=np.asarray(label)

    clf.fit(X, y)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(X[:,0],X[:,1])



    # w = clf.coef_[0]
    # a = -w[0] / w[1]
    # xx = np.linspace(-5, 5)
    # yy = a * xx - (clf.intercept_[0]) / w[1]
    #
    # # plot the parallels to the separating hyperplane that pass through the
    # # support vectors
    # b = clf.support_vectors_[0]
    # yy_down = a * xx + (b[1] - a * b[0])
    # b = clf.support_vectors_[-1]
    # yy_up = a * xx + (b[1] - a * b[0])
    #
    # # plot the line, the points, and the nearest vectors to the plane
    # plt.plot(xx, yy, 'k-')
    # plt.plot(xx, yy_down, 'k--')
    # plt.plot(xx, yy_up, 'k--')
    #
    # plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
    #             s=80, facecolors='none')
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    #
    # plt.axis('tight')
    # plt.show()


train_path = "/home/gaurav/PycharmProjects/Gait/train/"
test_path = "/home/gaurav/PycharmProjects/Gait/test/"


train_files = sorted(glob.glob(os.path.join(train_path , "*")))
test_files = sorted(glob.glob(os.path.join(test_path , "*")))
coord=[]

label=[]

for n,filename in enumerate(train_files):

    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 30, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
                x, y = i.ravel()
                x_y = []
                x_y.append(x)
                x_y.append(y)
                coord.append(x_y)
                label.append(n)


# print "forming cluster"
# print label.count(0),label.count(1),label.count(2),label.count(3)
cluster(coord,label)

test_x_y=[]

test_coord=[]
test_label=[]
accur_lin=[]


print "testing"
for n,filename in enumerate(test_files):
    img2 = cv2.imread(filename)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    corners2 = cv2.goodFeaturesToTrack(gray2, 30, 0.01, 10)
    corners2 = np.int0(corners2)

    for i in corners2:
                x, y = i.ravel()
                test_x_y = []
                test_x_y.append(x)
                test_x_y.append(y)
                test_coord.append(test_x_y)


                test_label.append(7)
    print "len is ",len(test_label)


    print clf.predict(test_coord)
    pred_lin = clf.score(test_coord, test_label)

    accur_lin.append(pred_lin)


print "accuract is ",accur_lin




    # print "count = ",count
    #result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst,Nonee()
    # # print dst.max(),dst.min()
    #
    #
    #
    # # Threshold for an optimal value, it may vary depending on the image.
    # img[dst>0.01*dst.max()]=[0,0,255]
    #
    # cv2.imshow('dst',img)
    # if cv2.waitKey(0) & 0xff == 27:

    #     cv2.destroyAllWindows()
