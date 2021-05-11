# -*- coding: utf-8 -*-
from src.main.com.wqj.cv.bighomework.extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("2021CVORDL\\") + len("2021CVORDL\\")]
# dataPath = rootPath + "Input/MLWorkHome/experiment3/data.txt"
"""
-query src\main\resources\com\wqj\cv\bighomework\database\674.jpg -index src/featureCNN.h5 -result src\main\resources\com\wqj\cv\bighomework\database
"""

ap = argparse.ArgumentParser()
ap.add_argument("-query", required=True,
                help="Path to query which contains image to be queried")
ap.add_argument("-index", required=True,
                help="Path to index")
ap.add_argument("-result", required=True,
                help="Path for output retrieved images")
args = vars(ap.parse_args())
args["index"] = rootPath + args["index"]
args["query"] = rootPath + args["query"]
args["result"] = rootPath + args["result"]

# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"], 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")

# read and show query image
queryDir = args["query"]
if (str(queryDir).endswith("png")):

    queryImg = mpimg.imread(queryDir,0)
else:
    queryImg = mpimg.imread(queryDir)
plt.figure(0)
plt.title("Query Image")
plt.imshow(queryImg)
plt.show()

# init VGGNet16 model
model = VGGNet()

# extract query image's feature, compute simlarity score and sort
queryVec = model.extract_feat(queryDir)
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]
# print rank_ID
# print rank_score



maxres = 3
imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:maxres])]
print("top %d images in order are: " % maxres, imlist)

# show top #maxres retrieved result one by one
for i, im in enumerate(imlist):
    if (str(queryDir).endswith("png")):
        image = mpimg.imread(args["result"] + "/" + str(im, encoding='utf-8'),0)
    else:
        image = mpimg.imread(args["result"] + "/" + str(im, encoding='utf-8'))
    plt.figure(i + 1)
    plt.title("search output %d" % (i + 1))
    plt.imshow(image)
    plt.show()
