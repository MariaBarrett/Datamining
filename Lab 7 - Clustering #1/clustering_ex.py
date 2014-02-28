from __future__ import division
import csv
import exceptions
import numpy as np
import random
import operator
from collections import Counter

irisfile = "iris.csv"

#global variables:
k = 3

def make_lists_of_attributes_from_file(filename):
	"""
	This function takes a filename of a csv-file and make a list of lists where each row is a datapoint.
	All values are converted from strings to a float.  
	The label is converted into 0.0, 1.0 and 2.0
	the dataset is shuffled before it's returned - but using the same shuffle every time
	"""
	filelist =[]

	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in reader:
			filelist.append(row)
	filelist = filelist[1:] #taking out the first row which is the header row
	for datapoint in filelist: 
		for n,i in enumerate(datapoint): 
			try:
				datapoint[n] = float(i) 

			except exceptions.ValueError:
				if 'setosa' in i:
					datapoint[n] = 0.0
				elif 'versicolor' in i:
					datapoint[n] = 1.0
				elif 'virginica' in i:
					datapoint[n] = 2.0
				else:
					print "unknown value", i
	random.shuffle(filelist, lambda:0.1)
	return filelist
dataset = make_lists_of_attributes_from_file(irisfile)

#Computing mean and variance
"""
This function takes a dataset and computes the mean and the variance of each input feature (leaving the class column out)
It returns two lists: [mean of first feature, mean of second feature] [variance of first feature, variance of second feature]
"""
def mean_variance(data):
	Mean = []
	Variance = []
	number_of_features = len(data[0]) - 1 #Leaving out the class
	for i in xrange(number_of_features): 
		s = 0
		su = 0

		#mean
		for elem in data:
			s +=elem[i]
		mean = s / len(data)
		Mean.append(mean)
		
		#variance:
		for elem in data:
			su += (elem[i] - Mean[i])**2
			variance = su/len(data)	
		Variance.append(variance)
	return Mean, Variance


"""
This function calls mean_variance to get the mean and the variance for each feature
Then these values are used to normalize every datapoint to zero mean and unit variance.
A copy of the data is created. 
The normalized values are inserted at the old index in the copy thus preserving class label 
The new, standardized data set is returned
"""
def meanfree(data):
	number_of_features = len(data[0]) - 1 #Leaving out the class
	mean, variance = mean_variance(data)

	new = np.copy(data)
	for i in xrange(len(new)):
		for num in xrange(number_of_features):
			#replacing at correct index in the copy
			new[i][num] = (new[i][num] - mean[num]) / np.sqrt(variance[num])
	return new 

def euclidean(ex1,ex2):
	"""
	This function takes two datapoints and calculates the euclidean distance between them.
	It expects a datapoint without class label in last position and a mean of features. 
	"""
	inner = 0
	for i in xrange(len(ex1)):
		inner += (ex1[i] - ex2[i])**2 
	distance = np.sqrt(inner)
	return distance

standardized_data = meanfree(dataset)


"""
This function expects a dataset and a list of centers. It calls the euclidean function and
stores the distance to each center in a list. From that list it gets the index of the minimum value
and stores the datapoint in a new list of lists at that index.
It returns the list of lists of the reordered datapoints

"""

def assigning_to_center(dataset, initial):
	sorted_according_to_center = [[],[],[]] 
	assert len(initial) == k
	
	for datapoint in dataset:
		distance = []
		for i in xrange(len(initial)):
			d = euclidean(datapoint[:-1], initial[i]) #leaving out the class before passing it on to the euclidean
			distance.append(d)
		min_index, min_value = min(enumerate(distance),key=operator.itemgetter(1)) #finding index of min value
		sorted_according_to_center[min_index].append(datapoint) #...and append in new list at that index

	return sorted_according_to_center

"""
This function expects a list of lists and calculates the mean (excluding the class label) for each list in the main list.
It returns a list of lists with the calculated means. 
"""

def gettingmean(sorted_according_to_center):
	listofmeans = []
	
	for sublist in sorted_according_to_center:
		temp = []
		for i in xrange(len(sublist[0])-1): 
			s = 0
			for elem in sublist:
				s +=elem[i]
			mean = s / len(sublist)
			temp.append(mean)
		listofmeans.append(temp)
	return listofmeans

#meanfree unit variance dataset 
standardized = meanfree(dataset)

"""
http://stackoverflow.com/questions/11601579/counter-most-common-elements-as-a-generator
"""
def most_common(iterable, n=None):
    return iter(Counter(iterable).most_common(n))

"""
Here is where everything is put together:
I have assigned 3 fixed datapoints to be my initial centers.
I assign datapoints to them and calculate mean1.
I assign to this mean and calculates mean1.

While mean1 and mean2 are different, I continue assigning and calculating new means

On the list of the separated datapoint I collect the label and count the most frequent label of each list
Then I calculate the accuracy with respect to that label. 
"""
def kmeans(standardized_data):
	initial = [standardized[10][:-1], standardized[100][:-1], standardized[50][:-1]] #leaving out the class

	sorted_to_center = assigning_to_center(standardized_data, initial)
	mean1 = gettingmean(sorted_to_center)

	sorted_to_center = assigning_to_center(standardized_data, mean1)
	mean2 = gettingmean(sorted_to_center)

	while mean1 != mean2:
		sorted_to_center = assigning_to_center(standardized_data, mean2)
		newmean2 = gettingmean(sorted_to_center)

		sorted_to_center = assigning_to_center(standardized_data, newmean2)
		newmean1 = gettingmean(sorted_to_center)

		mean1 = newmean1
		mean2 = newmean2

	labels = []
	for sublist in sorted_to_center:
		temp = []
		for elem in sublist:
			temp.append(elem[4])
		labels.append(temp)
	c = Counter()
	accuracy = 0
	for sublabellist in labels:
		for item in most_common(sublabellist,1):
			a = item[1] / len(sublabellist)
			accuracy += a
	accuracy = accuracy / k
	print "Accuracy:", accuracy


kmeans(standardized)





