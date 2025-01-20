import numpy as np
import struct

def load_dataset():
	with np.load("mnist.npz") as f:
		# convert from RGB to Unit RGB
		x_train = f['x_train'].astype("float32") / 255

		# reshape from (60000, 28, 28) into (60000, 784)
		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

		# labels
		y_train = f['y_train']

		# convert to output layer format
		y_train = np.eye(10)[y_train]

		return x_train, y_train

def load_test_dataset(images_path="t10k-images.idx3-ubyte", labels_path="t10k-labels.idx1-ubyte"):
    # Wczytywanie etykiet testowych
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        test_labels = np.fromfile(lbpath, dtype=np.uint8)

    # Wczytywanie obrazów testowych
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        test_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(test_labels), rows*cols)
        test_images = test_images.astype(np.float32) / 255.0

    # Kodowanie one-hot etykiet testowych
    one_hot_test_labels = np.zeros((test_labels.shape[0], 10))
    for i, label in enumerate(test_labels):
        one_hot_test_labels[i, label] = 1

    return test_images, one_hot_test_labels