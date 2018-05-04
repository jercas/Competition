"""
Created on Wed Mar 2 14:57:00 2018

@author: jercas
"""
import pandas as pd
import numpy as np
import time
import os

def predict(testPaths, predictions):
	print(testPaths)
	path = np.loadtxt("t.txt", dtype=np.str_, delimiter=' ')
	#for i in range(len(path)):
	#	predictions[i] = predictions[i].argmax(axis=0)
	output = pd.DataFrame(np.c_[path, np.zeros(path.shape)], columns=['path', 'label'])
	print(predictions)
	predictions = predictions.argmax(axis=1)
	print(predictions)

	for i, testPath in enumerate(testPaths):
		testPaths[i] = testPath.split(os.path.sep)[-1]
	print(testPaths)

	for i, pathDir in enumerate(testPaths):
		for j, pathTxt in enumerate(output.loc[:, 'path']):
			if str(pathDir) == str(pathTxt):
				output.loc[j, 'label'] = predictions[i]+1

	print(output)
	output.to_csv('./labeled/{}_{}_labeledTest.csv'.format(
				time.strftime('%Y-%m-%d',time.localtime(time.time())),
	            time.strftime('%H-%M-%S',time.localtime(time.time()))),
				sep=' ', header=False, index=False)
