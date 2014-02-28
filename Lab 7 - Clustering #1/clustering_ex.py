from __future__ import division
import csv
import exceptions
import numpy as np
import random

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
	for i in xrange(len(data)):
		for num in xrange(number_of_features):
			print num
			#replacing at correct index in the copy
			new[i][num] = (data[i][num] - mean[num]) / np.sqrt(variance[num])
	print new
	return new



def euclidean(ex1,ex2):
	"""
	This function takes two datapoints and calculates the euclidean distance between them. 
	Distances are in this way promoting closer neighbors.
	"""
	inner = 0
	for i in xrange(1, len(ex1)): #starting with 1, becuase index 0 is the class
		inner += (ex1[i] - ex2[i])**2 
	distance = np.sqrt(inner)
	return distance

standardized_data = meanfree(dataset)

def distance_to_center(dataset):
	initial = [dataset[0], dataset[1], dataset[2]]
	distance = []
	distances = []
	for k in initial:
		for d in dataset[:-1]: #not including the label 
			eu = euclidean(k,d)
			distance.append(eu)
		distances.append(sum(distance))
	return distances


distances = distance_to_center(dataset)
print distances
	




