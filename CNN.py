#Jay Rawal
#2017240

import torch
import numpy as np
from torchvision.transforms import *
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import sklearn.svm
import sklearn.model_selection
import sklearn.metrics
import sklearn.decomposition
import sklearn.multiclass

# constants
prob = 0.4
batch_size = 50
epochs = 10
l_rate = 0.001
label_number = 10

acc_training = []
cos_training = []
cos_training_epoch = []

trans_func = Compose([Resize(28, interpolation=2),RandomHorizontalFlip(p=0.5),ToTensor(),Normalize([0.], [1.])])

def getInputs():
	# MNIST Fashion
	l1 = dsets.FashionMNIST(root='FashionMNIST_data/',train=True,download=True,transform=trans_func)
	l2 = dsets.FashionMNIST(root='FashionMNIST_data/',train=False,download=True,transform=trans_func)
	l3 = torch.utils.data.DataLoader(dataset=l1,shuffle=True,batch_size=batch_size)
	l4 = torch.utils.data.DataLoader(dataset=l2,shuffle=True,batch_size=batch_size)
	return l1,l2,l3,l4

def getMean(x):
	return x.float().mean()  

def get_acc(desired_out, obtained_out):
	pred = obtained_out.data.max(dim=1)[1]
	accuracy = (pred.data == desired_out.data)
	accuracy = getMean(accuracy)  
	return accuracy.item()


def printy(s):
	# print("prob for Dropout ",prob)
	# print("batch size ",batch_size)
	# print("Number of iterations ",epochs)
	s+="Predicted"
	# print("learning rate ",l_rate,"\n")
	return None


class CNN(torch.nn.Module):

	def __init__(self):
		super(CNN, self).__init__()

		self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),torch.nn.ReLU())

		self.layer2 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=2),torch.nn.Dropout(p=prob))

		self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),torch.nn.ReLU())

		self.layer4 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),torch.nn.Dropout(p=prob))

		self.layer5 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
		torch.nn.init.xavier_uniform_(self.layer5.weight)
		
		self.layer6 = torch.nn.Linear(625, 10, bias=True)
		torch.nn.init.xavier_uniform_(self.layer6.weight)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(out.size(0), -1)   # Processing for Fully Connected Layer 
		out = self.layer5(out)
		out = self.layer6(out)
		return out


model = CNN()
SVM_model = sklearn.svm.SVC(kernel='rbf', class_weight='balanced')
criterion = torch.nn.CrossEntropyLoss()

def printConst():
	print("prob for Dropout ",prob)
	print("batch size ",batch_size)
	print("Number of iterations ",epochs)
	print("learning rate ",l_rate,"\n")

optimizer = torch.optim.Adam(params=model.parameters(), lr=l_rate)

printConst()
train_data,test_data,dataset_train,dataset_test = getInputs()
print("Size of the train dataset is ", train_data.data.size())
print("Size of the test dataset is ", test_data.data.size())

total_batch = len(train_data) // batch_size
print('Total number of batches is ',total_batch)

for iter in range(epochs):
	avg_cost = 0

	for i,(this_x, this_y) in enumerate(dataset_train):
		pl = 0
		optimizer.zero_grad()
		value = model(this_x)				#Predict Value
		pl+=(1+epochs)
		acc_training.append(get_acc(this_y, value))
		cost = criterion(value, this_y) 		# Loss func cost
		cost.backward() 					# Backward prop
		printy("Check")
		optimizer.step()
		pl=1
		cos_training.append(cost.item())
		
		avg_cost += cost.data / total_batch

		if (i % 50==0):
			print("iter=", iter+pl ,"\t batch = ", i ,"\t cost = ", cos_training[-1] ,"\t accuracy =",acc_training[-1])
			
			# break
	
	#SVM Epoch loss
	# outputs = model(iter(dataset_train)[:,0])
	# outputs_mlp = np.array(outputs.data)
	# labels = np.array(iter(dataset_train)[:,1])
	# SVM_model.fit(outputs_mlp,labels)
	
	# train_pred = SVM_model.predict(outputs_mlp)
	# train_score = SVM_model.score(train_pred,labels)
	
	# outputs = model(iter(dataset_test)[:,0])
	# outputs_mlp = np.array(outputs.data)
	# labels = np.array(iter(dataset_test)[:,1])

	# test_pred = SVM_model.predict(outputs_mlp)
	# test_score = SVM_model.score(test_pred,labels)
	# print("SVM : ",train_score,test_score)

	print("iter:",iter + 1,"averaged cost =",avg_cost.item())
	cos_training_epoch.append(avg_cost.item())

print('Learning Finished!')


plt.plot([i for i in range(len(acc_training))],acc_training)
plt.xlabel("Number of batches(all epochs)")
plt.ylabel("Accuracy")
plt.show()

plt.plot([i for i in range(len(cos_training))],cos_training)
plt.xlabel("Number of batches(all epochs)")
plt.ylabel("COST")
plt.show()

plt.plot([i for i in range(len(cos_training_epoch))],cos_training_epoch)
plt.xlabel("Number of epochs)")
plt.ylabel("COST")
plt.show()

#Testing data
correct = 0
total = 0

train_pred = []
train_req = []

confusion_matrix = torch.zeros(label_number, label_number)
with torch.no_grad():
	for data in dataset_train:
		pl=0
		test_x, labels = data
		outputs = model(test_x)
		printy("check")
		_, predicted = torch.max(outputs.data, 1)

		train_pred.append(outputs.data.numpy())
		train_req.append(labels.numpy())
		pl+=1
		for t_, p_ in zip(labels.view(-1), predicted.view(-1)):
			confusion_matrix[t_.long(), p_.long()] += 1
		
		pl = 0
		total += (labels.size(0)+pl)
		correct += (predicted == labels).sum().item()

print('\t\tTrain Data')
print("Confusion Matrix Training \n",confusion_matrix,"\n")
print('Accuracy of the network on the 60000 train data:', 100 * correct / total)

# print(train_req,train_pred)

#Testing data
correct = 0
total = 0

test_pred = []
test_req = []

confusion_matrix = torch.zeros(label_number, label_number)
with torch.no_grad():
	for data in dataset_test:
		test_x, labels = data
		outputs = model(test_x)
		_, predicted = torch.max(outputs.data, 1)

		# print(len(labels),len(outputs.numpy()[0]))
		test_pred.append(outputs.data.numpy())
		test_req.append(labels.numpy())
		
		for t_, p_ in zip(labels.view(-1), predicted.view(-1)):
			confusion_matrix[t_.long(), p_.long()] += 1
		
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print("Size of the test dataset is ", len(test_pred[-1]),len(test_pred[0]))


print('\t\tTesting Data')
print("Confusion Matrix Testing\n",confusion_matrix,"\n")
print('Accuracy of the network on the 10000 test data:', 100 * correct / total)

SVM_model = sklearn.svm.SVC(kernel='rbf', class_weight='balanced')

train_pred = np.array(train_pred).reshape(60000,10)
train_req = np.array(train_req).reshape(60000,1)

SVM_model.fit(train_pred,train_req)

y_pred = SVM_model.predict(train_pred)

print("\n SVM Confusion Matrix and Report Train ")
print(sklearn.metrics.classification_report(train_req, y_pred))
print(sklearn.metrics.confusion_matrix(train_req, y_pred))


test_pred = np.array(test_pred).reshape(10000,10)
test_req = np.array(test_req).reshape(10000,1)

y_pred = SVM_model.predict(test_pred)

print("\n SVM Confusion Matrix and Report Test ")
print(sklearn.metrics.classification_report(test_req, y_pred))
print(sklearn.metrics.confusion_matrix(test_req, y_pred))

"""
Refferences:

Basics
https://towardsdatascience.com/convolutional-neural-network-for-image-classification-with-implementation-on-python-using-pytorch-7b88342c9ca9

Transform
https://stackoverflow.com/questions/52120880/transforms-not-applying-to-the-dataset

Torch basics
https://pytorch.org/docs/stable/nn.html

Accuracy
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Confusion matrix
https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial

"""