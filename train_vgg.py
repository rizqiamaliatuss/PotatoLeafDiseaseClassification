#[PENTING] jika run pakai ini
# python train_vgg.py --dataset kentang --model output/finalprojectb1.h5 --label-bin output/finalprojectb1.pickle --plot output/finalprojectb1.png


# import matplotlib untuk simpan plot
import matplotlib
matplotlib.use("Agg")

# import semua library dan packages
from finalproject.vgg16 import VGG16
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# argumen parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# insialisasi data dan dan label
print("[INFO] loading images...")
data = []
labels = []

# randomly shuffle input dataset untuk membuat image path
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop input gambar
for imagePath in imagePaths:
	# load image path, kemudian resize ke 64x64 (VGG)
	# himpun data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	data.append(image)

	# ekstrak label class dari image path dan update
	# label list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# scale raw pixel intensities ke range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partisi dataset untuk training dan testing otomatis
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# konversi label dari integer ke vektor
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# augmentasi data dengan keras(otomatis)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize model
model = VGG16.build(width=64, height=64, depth=3,
	classes=len(lb.classes_))

# inisialisasi epoch, learning rate dan batch size
INIT_LR = 0.01
EPOCHS = 300
BS = 32

# inisialisasi model dan optimizer
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# training 
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# validasi
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))
print(confusion_matrix(testY.argmax(axis=1),
	predictions.argmax(axis=1)))

# plot training loss dan accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure(1)

# akurasi
plt.subplot(211) 
plt.plot(N, H.history["accuracy"], label="train_accuracy")
plt.plot(N, H.history["val_accuracy"], label="val_accuracy")
plt.title('model accuracy')  
plt.ylabel('Accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'validation'], loc='upper left')  

#loss
plt.subplot(212)
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title('model loss')  
plt.ylabel('Loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'validation'], loc='upper left')

#save plot
plt.tight_layout() 
plt.savefig(args["plot"])


# save model dan label
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
