import sys
import idx2numpy
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt


# acti_str = "linear"
# weight_c = 1
# batch_size = 2000
# learning_rate = 0.00000001
# epochs = 10

acti_str = "sigmoid"
weight_c = 0.1
batch_size = 2000
learning_rate = 0.01
epochs = 5


loss_array = []
y_len = 10
np.set_printoptions(threshold=sys.maxsize)

class Neural_Net():
	
	def __init__(self,n_layers,nodes_arr,acti_str,l_rate):
		self.layer_list =list()
		self.layer_count = n_layers
		self.acti_str = acti_str
		self.l_rate = l_rate

		bias = 0
		for i in range(1,self.layer_count):
			self.layer_list.append(Layer(nodes_arr[i],nodes_arr[i-1],bias))

	def score(self,X,Y):
		correct_num = self.predict(X,Y)
		ans = (correct_num/np.shape(X)[0])
		# print(ans)
		return ans

	def predict(self,X,Y):
		correct_num = 0
		ans =[]
		for i in range(np.shape(X)[0]):
			x = X[i]
			pred_out = self.forward_prop(x)
			ans.append(pred_out)
			y = np.argmax(pred_out)
			if y == Y[i]:
				correct_num+=1
		# print(ans)
		return correct_num

	def fit(self,nn_input,desired_out,btch_size,epochs):
		print("Fitting Training Data\n")
		X_train,Y_train = resample(nn_input, desired_out, n_samples=btch_size, stratify=desired_out)
		while(epochs>0):
			self.fit_help(X_train,Y_train)
			print("epoch = ",epochs," score = ",self.score(X_train,Y_train))
			epochs-=1

	def one_hot(self,y):
		hot_out = np.zeros(y_len)
		hot_out[y] = 1
		return hot_out

	def fit_help(self,X,Y):
		final_ys = []
		for i in range(np.shape(X)[0]):
			x = X[i]
			
			ans_x = []
			pre_ac = []
			ans_x.append(x)

			for j in range(self.layer_count-2):
				vj = np.dot(self.layer_list[j].weights,x)
				pre_ac.append(vj)
				x = self.activation_function(vj)
				ans_x.append(x)

			vj = np.dot(self.layer_list[-1].weights,x)

			final_ys.append(self.softmax(vj))
			ans_x.append(self.softmax(vj))
			
			self.back_prop(ans_x,pre_ac,Y[i])

		final_ys = np.array(final_ys)
		loss_array.append(self.loss(final_ys,Y))
	
	def printWeights(self):
		for i in range(self.layer_count-2):
			print(self.layer_list[i].weights)

	def forward_prop(self,x):
		for i in range(self.layer_count-2):
			x =  self.activation_function(np.dot(self.layer_list[i].weights,x))
		# return final_out
		return self.softmax(np.dot(self.layer_list[-1].weights,x))

	def back_prop(self,a,pre_act_out,y):
		list_del = []

		val_del = (a[-1] - self.one_hot(y))*self.softmax_grad(a[-1])
		list_del.append(val_del)

		for i in range(self.layer_count-3,-1,-1):
			temp1 = self.der_activation_function(a[i+1],pre_act_out[i])
			temp2 = np.dot(self.layer_list[i+1].weights.T,val_del)
			
			val_del = temp2*temp1
			list_del.append(val_del)

		list_del = list_del[::-1]
		for i in range(self.layer_count-1):
			self.layer_list[i].weights += self.l_rate*np.outer(list_del[i],a[i])


	def softmax(self,X):
		exps = np.exp(X - np.max(X))
		return exps / np.sum(exps)

	def loss(self,x,y):
		temp1 = -np.log(x[range(y.shape[0]),y]+ (10**-15))
		loss = np.sum(temp1) / (y.shape[0])
		return loss

	def activation_function(self,act_out):
		st=str.lower(self.acti_str)
		if(st=="relu"):
			act_out = np.where(act_out<0,0,act_out)
		elif(st=="sigmoid"):
			act_out = 1 / (1 + np.exp(-1*act_out))
		elif(st=="linear"):
			act_out = act_out
		elif(st=="tanh"):
			act_out = np.tanh(act_out)
		return act_out

	def softmax_grad(self,X):
		return (self.softmax(X)-1)/(X.shape[0])
	
	def der_activation_function(self,x,fx):
		st=str.lower(self.acti_str)
		ans = np.ones_like(x)
		if(st=="relu"):
			x[x<=0] = 0
			x[x>0] = 1
			ans = x
		elif(st=="sigmoid"):
			ans = fx*(1-fx)
		elif(st=="linear"):
			ans = np.ones_like(x)
		elif(st=="tanh"):
			ans = 1 - (fx*fx)
		return ans

class Layer():
	def __init__(self,my_node_count,last_node_count,bias):
		self.weights = np.random.randn(my_node_count,last_node_count,)*weight_c
		self.bias = bias

def takeInput():
	train_X = idx2numpy.convert_from_file('train-images.idx3-ubyte')
	train_Y = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

	test_X = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
	test_Y = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
	
	l = [train_X,train_Y,test_X,test_Y]
	
	return l


if __name__=="__main__":
	# no_layers = int(input("Number of layers : "))
	# layer_node_count = list(map(int,input("Individual layer node count : ").split(" ")))
	# learning_rate = float(input("Learning Rate : "))
	# input_arr = np.array(list(map(float,input("Input array : ").split(" "))))
	# acti_str = input("Activation Function : ")
	# desired_out = np.array(list(map(float,input("Desired Output : ").split(" "))))
	# epochs = int(input("No of iterations : "))

	l = takeInput()
	train_X,train_Y,test_X,test_Y = l[0],l[1],l[2],l[3]
	
	train_X = train_X.reshape(train_X.shape[0],28*28)
	test_X = test_X.reshape(test_X.shape[0],28*28)

	# print(train_X.shape,train_Y.shape,test_X,test_Y)

	NN = Neural_Net(6,[128,62,62,62,32,10],acti_str,learning_rate)
	NN.fit(train_X,train_Y,batch_size,epochs)
	
	# NN.printWeights()
	# print("\n\n\n")

	train_score = NN.score(train_X,train_Y)
	test_score = NN.score(test_X,test_Y)

	print("\nActivation Function ",acti_str)
	print("train_score Accuracy: ",train_score)
	print("test_score Accuracy: ",test_score)

	# stk = "Loss vs Epoc " + acti_str 
	# ep = [i for i in range(1,epochs+1)]
	# plt.plot(ep,loss_array)
	# plt.xlabel('Epocs')
	# plt.ylabel('Cross-Entropy Loss')
	# plt.title(stk)
	# plt.show()