from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import Counter
import random

#######################################################################
#																	  
#				Getting train and test sets from file
#												
#######################################################################
mainfile = open('agaricus-lepiotadata.txt', 'r')

"""
This function reads ind in the files, strips by newline and splits by comma char. 
Then it separates the data set into test and train sets. Every 4th goes to test, the rest goes to trains
"""
def file_to_train_and_test(filename):
	data_set = ([])
	for l in filename.readlines():
		l = l.rstrip('\n').split(',')
		data_set.append(l)
		data_set = data_set[::4]	
	#splitting into test and train sets. 
	trainset = []
	testset = []
	c = 4
	for elem in data_set:
		if c % 4 == 0:
			testset.append(elem)
			c+=1
		else:
			trainset.append(elem)
			c+=1
	return trainset, testset

trainset, testset = file_to_train_and_test(mainfile)

#######################################################################
#																	  
#						KNN
#												
#######################################################################

def Euclidean(ex1,ex2):
	"""
	This function takes two datapoints and calculates the euclidean distance between them. 
	It returns the squared distance, which is a way of weighing the neighbors. Distances are in this way promoting closer neighbors.
	"""
	inner = 0
	for i in range(len(ex1)-1): #We don't want the last value - that's the class
		inner += (ex1[i] - ex2[i])**2 
	distance = np.sqrt(inner)
	return distance**2

def hamming(ex1,ex2):
	assert (len(ex1) == len(ex2)),"Same lenght"
	counter = 0
	for c in xrange(1, len(ex1)): #starting with 1, becuase index 0 is the class
		if ex1[c] != ex2[c]:
			counter +=1
	return counter


def NearestNeighbor(tr,ex0,K):
	"""
  	This function expects a dataset, a datapoint and number of neighbors. 
  	It calls the euclidean and stores the distances with datapoint in a list of lists. 
  	These lists are sorted according to distances and K-nearest datapoints are returned 
	"""
	distances = []

	#distances.append(ex0)
	for ex in tr:
		curr_dist = hamming(ex,ex0) 
		distances.append([curr_dist,ex])

	distances.sort(key=itemgetter(0)) #sorting according to distance, shortest first
	KNN = distances[:K] #taking only the k-best matches
	without_distance = []
	for datapoint in KNN:
		without_distance.append(datapoint[1]) #At this index er get only the datapoint itself - not the distance.
	return without_distance

"""
This function calls KNN functions. I gets array (incl. class label) of KNN from NearestNeighbor-function. 
Most frequent class is counted. 
1-0 loss is calculated for using counter function. 

"""	
def eval(train,test,K):
	correct=0

	#test set		
	for ex in test:
		ex_prime=NearestNeighbor(train,ex,K)
		knn =[]
		for elem in ex_prime:
			knn.append(elem[0]) #that's the class
			result = Counter(knn)
		result = result.most_common(1)
		if result[0][0] == ex[0]: #comparing class labels
			correct +=1
	return correct/len(test)

#######################################################################
#																	  
#						ID3
#												
#######################################################################

trainset, testset = file_to_train_and_test(mainfile)

def train_id3(trainset):
	s = set() 
	setsofa = []
	for a in range(len(trainset[0])): #for every attribute
		partions_per_attribute = []
		for datapoint in trainset:
			temp = []
			temp.append(datapoint[a])
		s = set(temp)
	print s

train_id3(trainset)


"""
#Calling KNN and ID3
#Different K
K = [1,3,5]
k = 9
#Calling KNN
accuracy = eval(trainset, testset,k)
print "-"*45
print "Number of neighbors: \t%d" %k
print "Accuracy:\t%1.4f" %accuracy
print "-"*45
"""
