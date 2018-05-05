"""
Created on Sat May 5 13:16:00 2018

@author: jercas
"""

import sys
import pandas as pd
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from imutils import paths
import cv2

TEST_DIR = "./dataset/test_origin"
INPUTSHAPE = (224, 224)
MODEL = "modelAndWeights/{}".format("checkpoint-042-1.0920.hd5f")

DATA = pd.read_excel("label-name.xlsx")
classLabels = np.array(DATA.loc[:, "name"])
print(classLabels)

# Grab the list of images in the dataset then randomly sample indexes into the image path list.
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(TEST_DIR)))
idxs = np.random.randint(0, len(imagePaths), size=(20,))
imagePaths = imagePaths[idxs]

# Load the pre-trained model.
print("[INFO] loading pre-trained model")
model = load_model(MODEL)

# Loop over the sample images.
for (i, imagePath) in enumerate(imagePaths):
	image = load_img(imagePath, target_size=INPUTSHAPE)
	image = img_to_array(image)
	image = image.astype("float") / 255.0
	image = np.expand_dims(image, axis=0)
	# Predict
	print("[INFO] predicting")
	preds = model.predict(image)
	# Load the test example image, draw the prediction, and display it to screen.
	image = cv2.imread(imagePath)
	cv2.putText(image, u"Label: {}-{}".format(str(preds.argmax(axis=1)+1), classLabels[preds.argmax(axis=1)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)