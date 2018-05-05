# -*- coding: utf-8 -*-
"""
Created on Fri May 3 15:20:00 2018

@author: jercas
"""
from imutils import paths
import pandas as pd
import numpy as np
from icecream.icecream import ic
import shutil
import os

pathDirs = list(paths.list_images("./dataset/train_origin"))
data = np.loadtxt("train.txt", dtype=np.str_, delimiter=' ')
pathTxts = data[:, 0]
pathLabel = data[:, 1]
count = 1

for i, pathTxt in enumerate(pathTxts):
	for j, pathDir in enumerate(pathDirs):
		purePathDir = pathDir.split(os.path.sep)[-1]
		if pathTxt == purePathDir:
			ic(pathDir)
			ic(pathTxt)
			label = pathLabel[i]
			if not os.path.exists("./dataset/train_split/{}".format(label)):
				os.mkdir("./dataset/train_split/{}".format(label))
			shutil.copyfile(pathDir, "./dataset/train_split/{}/{}".format(label, pathTxt))
			ic(count)
			count+=1