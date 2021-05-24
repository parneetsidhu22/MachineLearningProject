import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


CATEGORIES = ["With Mask", "Without Mask"]


def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return [new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1),img_array]  # return the image with shaping that TF wants.

model = tf.keras.models.load_model("Model-99.71")

running = True

while running:
	image_name = input("Enter the file path: ")
	if image_name == 'exit()' or image_name == "quit()":
		running = False
	else:
		try:
			img = prepare(image_name)

			prediction = model.predict([img[0]])
			prediction = CATEGORIES[int(prediction[0][0])]

			if prediction == "Without Mask":
				color = "red"
			else:
				color = "green"


			plt.imshow(img[1],cmap="gray")
			plt.text(20,0,f"Prediction: {prediction}",bbox=dict(fill=False, edgecolor=color, linewidth=2), color=color, size=20)
			plt.show()
		except IOError as e:
			print("File path specified does not exists.")