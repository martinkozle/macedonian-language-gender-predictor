from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import numpy as np
import io
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


class Model:
	def __init__(self, X_train, X_test, Y_train, Y_test):
		self.X_train = np.array(X_train)
		self.X_test = np.array(X_test)
		self.Y_train = np.array(Y_train)
		self.Y_test = np.array(Y_test)
		# define the keras model
		self.model = Sequential()
		self.model.add(Dense(50, input_dim=25, activation='relu'))
		self.model.add(Dense(150, activation='relu'))
		self.model.add(Dense(50, activation='relu'))
		self.model.add(Dense(3, activation='sigmoid'))
		# compile the keras model
		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	def train(self, epochs, batch_size):
		print("Training.")
		# fit the keras model on the dataset
		self.model.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size)
		print("Training finished.")

	def eval(self):
		print("Evaluating.")
		# evaluate the keras model
		_, accuracy = self.model.evaluate(self.X_test, self.Y_test)
		print('Accuracy: %.2f' % (accuracy * 100))

	def load(self, name):
		self.model = load_model("models/" + name + ".h5")
		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		self.model.summary()
		print("Loaded model from disk")

	def save(self, name):
		self.model.save("models/" + name + ".h5")
		print("Saved model to disk")

	def predict(self, word):
		arr, _ = scaleData([word], [])
		arr = np.array(arr)
		print("Prediction: ", end='')
		prediction = self.model.predict(arr)[0]
		if max(prediction) == prediction[0]:
			print("м")
		elif max(prediction) == prediction[1]:
			print("ж")
		else:
			print("с")


def getData(fileName):
	words = []
	labels = []
	file = io.open(fileName, mode="r", encoding="utf-8")
	data = file.read().split('\n')
	for i in data:
		temp = i.split(',')
		words.append(temp[0])
		labels.append(temp[1])
	return words, labels


def scaleData(X, Y):
	for i in range(len(X)):
		X[i] = " " * (25 - len(X[i])) + X[i]
	labelMap = {'м': (1, 0, 0), 'ж': (0, 1, 0), 'с': (0, 0, 1)}
	macedonianCharacters = " -’абвгдѓежзѕијклљмнњопрстќуфхцчџшѝѐ"
	dataMap = dict(zip(list(macedonianCharacters), range(0, 36)[::-1]))
	# print(value)
	scaledData = []
	tupleLabels = []
	for word in X:
		temp = []
		for char in word:
			# scaled_value = (value - min) / (max - min)
			temp.append((dataMap[char.lower()] - 36) / (-36))
			# I messed up the formula ^ but it still works so mehh?
		scaledData.append(temp)
	for gender in Y:
		tupleLabels.append(labelMap[gender])
	return scaledData, tupleLabels


def splitData(X, Y):
	return train_test_split(X, Y, test_size=0.3)


def main():
	words, labels = getData("data.txt")
	print("Gathered data.")
	X, Y = words, labels
	X_scale, Y = scaleData(X, Y)
	print("Scaled all the data.")
	X_train, X_test, Y_train, Y_test = splitData(X_scale, Y)
	model = Model(X_train, X_test, Y_train, Y_test)
	if input("Do you want to load a model file? ") in ("y", "yes", "yea", "da"):
		model.load(input("Input file name (without extenstion): "))
		model.eval()
	else:
		epochs = int(input("Enter number of epochs: "))
		batch_size = int(input("Enter batch size: "))
		model.train(epochs, batch_size)
		model.eval()
		if input("Do you want to save this model? ") in ("y", "yes", "yea", "da"):
			model.save(input("Input file name (without extenstion): "))
	while True:
		model.predict(input("Test with manual input: "))


# learn(words, labels)


if __name__ == "__main__":
	main()
