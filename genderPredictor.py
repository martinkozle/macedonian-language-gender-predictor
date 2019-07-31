import tensorflow as tf
from sklearn.model_selection import train_test_split
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import io


class Model:
	def __init__(self, X_train, X_test, Y_train, Y_test):
		self.X_train = X_train
		self.X_test = X_test
		self.Y_train = Y_train
		self.Y_test = Y_test
		# define the keras model
		self.model = Sequential()
		self.model.add(Dense(30, input_dim=20, activation='relu'))
		self.model.add(Dense(15, activation='relu'))
		self.model.add(Dense(3, activation='sigmoid'))

	def train(self):
		print("Training.")
		# compile the keras model
		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		# fit the keras model on the dataset
		self.model.fit(self.X_train, self.Y_train, epochs=150, batch_size=10)
		print("Training finished.")

	def eval(self):
		print("Evaluating.")
		# evaluate the keras model
		_, accuracy = self.model.evaluate(self.X_test, self.Y_test)
		print('Accuracy: %.2f' % (accuracy * 100))

	def predict(self, word):
		print("Prediction: " + self.model.predict_classes(scaleData([word])[0]))


def getData(fileName):
	words = []
	labels = []
	file = io.open(fileName, mode="r", encoding="utf-8")
	data = file.read().split('\n')
	for i in data:
		temp = i.split(',')
		words.append(temp[0])
		labels.append(temp[1])
	for i in range(len(words)):
		words[i] = " " * (20 - len(words[i])) + words[i]
	return words, labels


def scaleData(X, Y):
	labelMap = {'м': (1, 0, 0), 'ж': (0, 1, 0), 'с': (0, 0, 1)}
	macedonianCharacters = " -’абвгдѓежзѕијклљмнњопрстќуфхцчџшѝѐ"
	dataMap = dict(zip(list(macedonianCharacters), range(0, 36)[::-1]))
	# print(value)
	scaledData = []
	tupleLabels = []
	for word in X:
		temp = []
		for char in word:
			temp.append((dataMap[char.lower()] - 36) / (-36))
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
	model.train()
	model.eval()
	while True:
		model.predict(input("Test with manual input: "))


# learn(words, labels)


if __name__ == "__main__":
	main()
