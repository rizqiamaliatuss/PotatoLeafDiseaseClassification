# [Penting] untuk run program
# python classify.py --image images/alter.jpg --model output/finalprojectb1.model --label-bin output/finalprojectb1.pickle --width 64 --height 64

# import library dan packages
from keras.models import load_model
import argparse
import pickle
import cv2

# argumen parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image we are going to classify")
ap.add_argument("-m", "--model", required=True,
	help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to label binarizer")
ap.add_argument("-w", "--width", type=int, default=28,
	help="target spatial dimension width")
ap.add_argument("-e", "--height", type=int, default=28,
	help="target spatial dimension height")
ap.add_argument("-f", "--flatten", type=int, default=-1,
	help="whether or not we should flatten the image")
args = vars(ap.parse_args())

# load gambar dan resize
image = cv2.imread(args["image"])
output = image.copy()
image = cv2.resize(image, (args["width"], args["height"]))

# scale pixel values to [0, 1]
image = image.astype("float") / 255.0

# flatten cek
if args["flatten"] > 0:
	image = image.flatten()
	image = image.reshape((1, image.shape[0]))

else:
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))

# load model dan label
print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# prediksi
preds = model.predict(image)
print(preds)

# menentukan class label untuk gambar
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# munculin hasil klasifikasi + prosentase
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (5, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.7,
	(0, 0, 255), 2)


# munculin gambar
cv2.imshow("Potato Leaf Disease Classification Based on Deep Learning", output)
cv2.waitKey(0)
