from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import Counter
import random
import math

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
	data_set = data_set[::20] # for development - taking onle every 4th

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
	mainfile.close()
	return trainset, testset

#Calling function to create train and test set
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
	for i in xrange(1, len(ex1)): #starting with 1, becuase index 0 is the class
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

	for ex in test:
		ex_prime=NearestNeighbor(train,ex,K)
		knn =[]
		for elem in ex_prime:
			knn.append(elem[0]) #that's the class being appended to a list
			result = Counter(knn) 
		result = result.most_common(1) # counting the most frequent among the votes
		if result[0][0] == ex[0]: #comparing class labels
			correct +=1
	return correct/len(test)

#######################################################################
#																	  
#						ID3
#												
#######################################################################
"""
This function is from http://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
It returns true if all values in a list are equal
"""
def checkEqual1(iterator):
  try:
     iterator = iter(iterator)
     first = next(iterator)
     return all(first == rest for rest in iterator)
  except StopIteration:
     return True

def entropy(train):
	temp = []
	labels = []
	for datapoint in train:
		labels.append(datapoint[0])
	counted = Counter(labels)
	ent=0.0
	for key in counted.keys():
		prob=float(counted[key])/len(train)
		ent+= -prob*math.log(prob,2)
	return ent

def splitDataset(dataset,nclom,value):
    retDataSet=[]
    for record in dataset:
        if record[nclom] == value:
            reducedRecord=record[:nclom]
            reducedRecord.append(record[nclom+1:])
            retDataSet.append(reducedRecord)
    return retDataSet

def set_of_attributes(data):
	s = set() 
	setsofa = []
	for a in xrange(1, len(data[0])): #starting with 1, becuase index 0 is the class
		temp = []
		for datapoint in data:
			temp.append(datapoint[a]) #appending the a'th attribute of every datapoint			
		setoftemp = set(temp)
		setsofa.append(setoftemp) #setsofa is a list of the sets for all attributes i.e. the possible outcomes for each attribute
	return setsofa

def findbestsplit(trainset):
	traincopy = np.copy(trainset)
	root_entropy = entropy(traincopy)
	bestInfoGain = 0.0
	bestFeature = -1

	setsofa = set_of_attributes(traincopy)

	for attrindex in xrange(1, len(traincopy)):
		for val in setsofa[attrindex]:
			subDataset = splitDataset(traincopy, attrindex, val)
			prob=len(subDataset)/float(len(trainset))
        	new_entropy += prob*entropy(subDataset)
   		infoGain=root_entropy-new_entropy
    	if infoGain > bestInfoGain:
        		bestInfoGain=infoGain
        		bestFeature=i
	print bestFeature
	return bestFeature

findbestsplit(trainset)
"""
def buildTree(dataset):
	classes_in_branch= []
	for a in xrange(len(traincopy)): 
		classes_in_branch.append(traincopy[a][0]) #the class of each datapoint
	if checkEqual1(classes_in_branch) == True: #check if all classes in a branch are the same
		label = classes_in_branch[0]
	#if countbranches => len(traincopy[0]): #if there are no more attributes to split by
	#	c = Counter(countbranches)	
	#	result = c.most_common(1)
	#	label = result[0]

	classlist=[ x[0] for x in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(classlist)==1:
    	print "here goes the most frequent class"
        #return majorityCnt(classlist)
    bestFeature=chooseBestFeatureToSplit(dataset)
    bestFeatureLabel=labels[bestFeature]
    tree={bestFeatureLabel:{}}
    del(labels[bestFeature])
    featValues = [x[bestFeature] for x in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        tree[bestFeatureLabel][value] = buildTree(splitDataset(dataset, bestFeature, value),subLabels)
	return tree
 
	if traincopy == []:
		print "no more samples to train on"

 
def chooseBestFeatureToSplit(dataset):
    baseEntropy=entropy(dataset)
    bestInfoGain=0.0
    bestFeature=-1

    for i in xrange(1,len(dataset[0])):
       	featureList=[x[i] for x in dataset]
        uniqueValues=set(featureList)
        newEntropy=0.0
        for value in uniqueValues:
            subDataset=splitDataset(dataset, i, value)
            prob=len(subDataset)/float(len(dataset))
            newEntropy += prob*entropy(subDataset)
        infoGain=baseEntropy-newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def buildTree(dataset):
    classlist=[ x[-1] for x in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(classlist)==1:
        classlist.sorted
        #return majorityCnt(classlist)
    bestFeature=chooseBestFeatureToSplit(dataset)
    bestFeatureLabel=labels[bestFeature]
    tree={bestFeatureLabel:{}}
    del(labels[bestFeature])
    featValues = [x[bestFeature] for x in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        tree[bestFeatureLabel][value] = buildTree(splitDataset(dataset, bestFeature, value),subLabels)
	return tree
 
tree = buildTree(trainset)

#Calling KNN

K = [1,3,5]
k = 9
#Calling KNN
accuracy = eval(trainset, testset,k)
print "-"*45
print "Number of neighbors: \t%d" %k
print "Accuracy:\t%1.4f" %accuracy
print "-"*45
"""

