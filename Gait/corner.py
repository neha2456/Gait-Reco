import cv2
import numpy as np
import glob
import os

import numpy as np

from dtk.inertia import y_rot
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt


from matplotlib import style


def cluster(X,Y):
    style.use("ggplot")




    plt.scatter(X, Y)
    plt.show()

    ms = MeanShift()
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    print(cluster_centers)
    n_clusters_ = len(np.unique(labels))
    print("Number of estimated clusters:", n_clusters_)

    colors = 10 * ['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']

    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                marker="x", color='k', s=150, linewidths=5, zorder=10)

    plt.show()


faces_folder_path = "/home/gaurav/PycharmProjects/Gait/00_4/"


files = sorted(glob.glob(os.path.join(faces_folder_path , "*")))

x_coord=[]
y_coord=[]
for filename in files:
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    a = np.zeros(shape=(240, 352))
    for i in range(0,240):
        for j in range(0,352):
            if dst[i][j]>0.01*dst.max():
                x_coord.append(i)
                y_coord.append(j)




    #
    # print "count = ",count
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # print dst.max(),dst.min()
    cluster(x_coord,y_coord)


    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()