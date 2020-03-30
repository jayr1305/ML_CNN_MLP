import idx2numpy
import numpy as np
from sklearn.neural_network import MLPClassifier

def takeInput():
	train_X = idx2numpy.convert_from_file('train-images.idx3-ubyte')
	train_Y = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

	test_X = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
	test_Y = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
	
	l = [train_X,train_Y,test_X,test_Y]
	
	return l

if __name__=="__main__":

	acti_str = "relu"
	learning_rate = 0.1
	epochs = 10

	l = takeInput()
	train_X,train_Y,test_X,test_Y = l[0],l[1],l[2],l[3]
	
	train_X = train_X.reshape(train_X.shape[0],28*28)
	test_X = test_X.reshape(test_X.shape[0],28*28)

	mlp = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=epochs,solver='adam', verbose=True)

	mlp.fit(train_X, train_Y)
	print("Training set score: " ,mlp.score(train_X, train_Y))
	print("Test set score: " ,mlp.score(test_X, test_Y))