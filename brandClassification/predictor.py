"""
Created on Wed Mar 2 14:57:00 2018

@author: jercas
"""
import pandas as pd
import numpy as np
import os

def predict(testPaths, predictions):
	path = np.loadtxt("test.txt", dtype=np.str_, delimiter=' ')
	output = pd.DataFrame(np.c_[path, np.zeros(path.shape)], columns=['path', 'label'])
	predictions = predictions.argmax(axis=1)

	for i, path in enumerate(testPaths):
		testPaths[i] = path.split(os.path.sep)[-1]
	print(testPaths)

	for i, pathDir in enumerate(testPaths):
		for j, pathTxt in enumerate(output.loc[:, 'path']):
			if pathDir == pathTxt:
				output.loc[j, 'label'] = predictions[i]

	print(output)
	output.to_csv('labeledTest.csv', header=0)


