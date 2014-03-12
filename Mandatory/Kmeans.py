from __future__ import division
import exceptions
import numpy as np
import random
import operator
from collections import Counter

#global variables:
k = 2

def euclidean(ex1,ex2):
	"""
	This function takes two datapoints and calculates the euclidean distance between them.
	It expects two data points without class label 
	"""
	assert len(ex1) == len(ex2)
	inner = 0
	for i in xrange(len(ex1)):
		inner += (ex1[i] - ex2[i])**2 
	distance = np.sqrt(inner)
	return distance

"""
This function expects a dataset and a list of centers. It calls the euclidean function and
stores the distance to each center in a list. From that list it gets the index of the minimum value
and stores the datapoint in a new list of lists at that index.
It returns the list of lists of the reordered datapoints

"""

def assigning_to_center(dataset, initial):
	sorted_according_to_center = [[],[]] 
	assert len(initial) == k
	
	for datapoint in dataset:
		distance = []
		for i in xrange(len(initial)):
			d = euclidean(datapoint, initial[i]) 
			distance.append(d)
		min_index, min_value = min(enumerate(distance),key=operator.itemgetter(1)) #finding index of min value
		sorted_according_to_center[min_index].append(datapoint) #...and append in new list at that index

	return sorted_according_to_center

"""
This function expects a list of lists and calculates the mean for each list in the main list.
It returns a list of lists with the calculated means. 
"""

def gettingmean(sorted_according_to_center):
	listofmeans = []
	for sublist in sorted_according_to_center:
		submean = sum(sublist) / len(sublist)
		listofmeans.append(submean)
	return listofmeans

"""
Here is where everything is put together:
I have assigned 3 fixed datapoints to be my initial centers.
I assign datapoints to them and calculate mean1.
I assign to this mean and calculates mean1.

While mean1 and mean2 are different, I continue assigning and calculating new means
"""
def kmeans(standardized_data):
	initial = [standardized_data[6], standardized_data[7]]

	sorted_to_center = assigning_to_center(standardized_data, initial)
	mean1 = gettingmean(sorted_to_center)

	sorted_to_center = assigning_to_center(standardized_data, mean1)
	mean2 = gettingmean(sorted_to_center)
	for i in xrange(k):
		while len(set(mean1[i]).intersection(mean2[i])) < len(mean1[i]): #
		#while all(mean1[i]) != all(mean2[i]):
			sorted_to_center = assigning_to_center(standardized_data, mean2)
			newmean2 = gettingmean(sorted_to_center)

			sorted_to_center = assigning_to_center(standardized_data, newmean2)
			newmean1 = gettingmean(sorted_to_center)

			mean1 = newmean1
			mean2 = newmean2
	print "*" * 45
	print "K-means clustering"
	print "*" * 45
	print "k = ",k
	print "Mean of clusters:", mean1






