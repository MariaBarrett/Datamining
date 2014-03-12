from __future__ import division
import numpy as np
from operator import itemgetter
from collections import Counter

#-----------------------------------------------------------------------


def Euclidean(ex1,ex2):
	"""
	This function takes two datapoints and calculates the euclidean distance between them 
	excluding the last value, which is the class label. 
	It returns the distance.
	"""
	inner = 0
	for i in range(len(ex1)-1): #We don't want the last value - that's the class
		inner += (ex1[i] - ex2[i])**2 
	distance = np.sqrt(inner)
	return distance



def NearestNeighbor(train,ex0,K):
	"""
  	This function expects a dataset, a datapoint and number of neighbors. 
  	It calls the euclidean and stores the distances with datapoint in a list of lists. 
  	These lists are sorted according to distances and K-nearest datapoints are returned 
	"""
	distances = []

	for ex in train:
		curr_dist = Euclidean(ex,ex0) 
		distances.append([curr_dist,ex])

	distances.sort(key=itemgetter(0))
	KNN = distances[:K] #taking only the k-best matches
	return KNN



"""
This function calls KNN functions. I gets array (incl. class label) of KNN from NearestNeighbor-function. 
Most frequent class is counted. 
1-0 loss and accuracy is calculated for train and test using counters. 
For the train accuracy I train on train and use datapoints from the same set.
For the test acc I train on train and use datapoints from test. 
"""	
def eval(train, test, K):
	wrongtest=0

	#test set		
	for ex in test:
		ex_prime=NearestNeighbor(train,ex,K)
		knn =[]
		for elem in ex_prime:
			knn.append(elem[-1][-1]) #that's the class
			result = Counter(knn)
		result = result.most_common(1)
		if result[0][0] != ex[-1]:
			wrongtest +=1
	return wrongtest/len(test)



