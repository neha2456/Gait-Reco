import glob
import os

test_path = "/home/gaurav/PycharmProjects/Gait/test_exact/"

test_files = sorted(glob.glob(os.path.join(test_path , "*")))

for n,filename in enumerate(test_files):
    print filename
    # comp_path = filename +"/"
    # comp_files = sorted(glob.glob(os.path.join(comp_path, "*")))
    # for x in comp_files:
    #
    #     print x
