"""
Created on Wed Mar 2 10:00:00 2018

@author: jercas
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from keras.layers import GlobalAvgPool2D, Flatten, Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19

from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import matplotlib
# Set matplotlib backend to Agg to indicate to create a non-interactive figure that will simply be saved to disk.
# Why? ^^^
# Depending on what your default matplotlib backend is and whether you are accessing your deep learning machine remotely
#(via SSH, for instance), X11 session may timeout. If that happens, matplotlib will error out when it tries to display figure.
matplotlib.use("Agg")
from preprocessing.SimplePreprocessor import SimplePreprocessor
from preprocessing.ImageToArrayPreprocessor import ImageToArrayPreprocessor
from preprocessing.SimpleDatasetLoader import SimpleDatasetLoader
from callbacks.trainingMonitor import TrainingMonitor
from stepBased_lr_decay import stepBased_decay
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import os

# Define a dictionary that maps model names to its classes inside keras individual.
MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception,
	"resnet": ResNet50
}
CLASSES = 100
BATCH_SIZE = 64
EPOCHS = 50

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--monitor", required=False,
                help="decide whether to use training monitor which can plot loss curve at the end of every epoch.")
ap.add_argument("-o", "--model", type=str, default="vgg16", help="name of pre-trained network to use")
ap.add_argument("-c", "--checkpoint", required=False,
                help="decide whether to store checkpoint which can serialized models during the training process on each improvement epoch.")
args = vars(ap.parse_args())

# Show information on the process ID.
print("[INFO] process ID: {}".format(os.getpid()))

# Ensure a valid model name was supplied via command line argument.
if args["model"] not in MODELS.keys():
	raise AssertionError("The --model command line argument should be a key in the 'MODELS' dictionary which include VGG16, VGG19, 1ncepthonV3, Xception, Resnet50")

imagePaths = list(paths.list_images(args["dataset"]))
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception", "xception"):
	inputShape = (299, 299)
	# Updating preprocess to use a separate pre-processing function that performs a different type of scaling since the
	#Inception and its extension Xception is act as "multi-level feature extractor" by computing different size convolution
	#filters within the same module of the network.
	preprocess = preprocess_input

# Load the input image using the keras helper utility while ensuring the image is resized to 'inputShape', the required
#input dimensions for ImageNet pre-trained network.
print("[INFO] loading and pre-processing image...")
# Initialize the image preprocessors.
sp = SimplePreprocessor(inputShape[0], inputShape[1])
iap = ImageToArrayPreprocessor()
# Load the dataset from disk then scale the raw pixel RGBs to the range [0, 1].
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(trainX, trainY) = sdl.load("{}/train".format(imagePaths), verbose=500)
trainX = trainX.astype("float") / 255.0
(valX, valY) = sdl.load("{}/val".format(imagePaths), verbose=500)
valX = valX.astype("float") / 255.0
(testX, testY) = sdl.load("{}/test".format(imagePaths), verbose=1000)
testX = testX.astype("float") / 255.0

# One-hot encoding: convert the labels from integers to vectors.
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
valY = lb.fit_transform(valY)
testY = lb.fit_transform(testY)

# Initialize the optimizer and model.
print("[INFO] compiling model...")
"""
# Time-based decay: slowly reduce the learning rate over time, common setting for decay is to divide the initial lr
#by total number of epochs.(here 0.01/40)
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
"""

# Step-based decay: alpha = initialAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
# lr parameter can be leave out entirely since it is using the LearningRateScheduler callback.
opt = SGD(momentum=0.9, nesterov=True)

# Define the set of callbacks function to be passed to the model during training.
# Keras will call callbacks at the start or end of every epoch, mini-batch update, etc.
# Then 'LearningRateSchedular' will call 'stepBased_decay' at the end of every epoch, decide whether to update learning 
#rate prior to the next epoch starting. 
callbacks = [LearningRateScheduler(stepBased_decay)]

# Load network weights from disk.
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
# Adaptive function call.
model = Network(weights="imagenet", include_top=False, classes=100)
# Freeze original layer weights.
for layer in model.layers:
	layer.trainable = False

model.add(GlobalAvgPool2D(name="classify_avgPooling"))
model.add(Flatten(input_shape=model.output_shape[1:]))
model.add(Dense(256, activation='relu'), name="fc1")
model.add(Activation("relu"))
model.add(BatchNormalization(name="classify_bn1"))
model.add(Dropout(0.5))
model.add(Dense(CLASSES), name="fc2")
model.add(Activation("softmax", name="predictions"))

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

if args["monitor"]:
	if not os.path.exists("{}".format(args["monitor"])):
		os.mkdir("{}".format(args["monitor"]))
	if not os.path.exists("{}/{}".format(args["monitor"], os.getpid())):
		os.mkdir("{}/{}".format(args["monitor"], os.getpid()))

	print("\n[INFO] monitor module establish!")
	figurePath = "{}/{}".format(args["monitor"], os.getpid())
	jsonPath   = "{}/{}/{}.json".format(args["monitor"], os.getpid(), os.getpid())
	# Construct the set of callbacks.
	callbacks.append(TrainingMonitor(figurePath=figurePath, jsonPath=jsonPath))

if args["checkpoint"]:
	if not os.path.exists("{}".format(args["checkpoint"])):
		os.mkdir("{}".format(args["checkpoint"]))
	if not os.path.exists("{}/{}".format(args["checkpoint"], os.getpid())):
		os.mkdir("{}/{}".format(args["checkpoint"], os.getpid()))

	print("\n[INFO] checkpoint module establish!\n")
	# A template string value that keras uses when writing checkpoing-models to disk based on its epoch and the validation
	#value on the current epoch.
	fname = os.path.join("{}/{}/".format(args["checkpoint"], os.getpid()), "checkpoint-{epoch:03d}-{val_loss:.4f}.hd5f")
	# monitor -- what metric would like to monitor;
	# mode -- controls whether the ModelCheckpoint be looking for values that minimize metric or maximize it in the contrary.
	#         such as, if you monitor val_loss, you would like to minimize it and if monitor equals to val_acc then you should maximize it.
	# save_best_only -- ensures the latest best model (according to the metric monitored) will not be overwritten.
	# verbose=1 -- simply logs a notification to terminal when a model is being serialized to disk during training.
	# period -- the interval epochs between two saved checkpoints.
	checkpoint = ModelCheckpoint(filepath=fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
	# Construct the set of callbacks.
	callbacks.append(checkpoint)

# Train.
print("[INFO] training network...")
Hypo = model.fit(trainX, trainY, validation_data=(valX, valY), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=callbacks)

# Evaluate.
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))

# Plot the training loss, training accuracy, validation loss, validation accuracy over time.
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), Hypo.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), Hypo.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), Hypo.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), Hypo.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("{}.png".format(args["output"]))
plt.show()
