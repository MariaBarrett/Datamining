from __future__ import division
import numpy as np
import csv
from random import choice
from collections import Counter
import itertools
import copy
import Apriori # my own apriori code
import Kmeans #my own K-means code
import KNN #my own K-Nearest Neighbor

###########################################################################################
#
#									Main file
#
###########################################################################################

#Opening file, stripping and splitting. The structure is as follows [header[first header][second header]...][second person[first answer]...]
#Attributes variables

file2014 = "data_mining_2014_dataset.csv"

def make_lists_of_attributes_from_file(filename):
	"""
	This function takes a filename of a csv-file and make a list of lists of all column (per attribute). 
	The structure is as follows: 
	[[label of attribute 1, first persons answer to attribute 1, second persons answer...][label of attribute 2, first persons answer to attribute 2, second...]...]
	This list of lists is returned
	"""
	attribute =[]
	listofattributes = []

	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=';', quotechar='|')
		for row in reader:
			lines = [l.strip().split(';') for l in row]
			attribute.append(lines)
		temp=[]
		for x in xrange(len(attribute[1])): #number of features
			for i in xrange(len(attribute)): #number of persons
				temp.append(attribute[i][x][0])

			listofattributes.append(temp)
			temp=[]
	return listofattributes

###########################################################################################
#
#									Preprocessing
#
###########################################################################################

"""
This function tests if a string is a number.
"""
def is_number(some_chars):
    try:
        float(some_chars)
        return True
    except ValueError:
        return False

"""
This function expects the cleaned data and describes through 
mode, mean, variance, max, min, median and 2nd and 3rd quartiles
"""
def describing(cleaned_data):

	accummulated = 0

	#mode
	counted = Counter(cleaned_data)
	typevaltuple = counted.most_common(1)
	typeval = typevaltuple[0][0]

	#mean
	for elem in cleaned_data:
		accummulated += elem
	mean = accummulated / len(cleaned_data)

	#Variance
	accummulated = 0
	for elem in cleaned_data:
		accummulated = (elem)**2
	variance = (accummulated - (mean)**2 / len(cleaned_data)) / len(cleaned_data)

	copy=np.copy(cleaned_data)
	copy.sort() #for the following descriptors the data is sorted lowest to highest
	
	#max, min
	maxval = (copy[-1])
	minval = (copy[0])

	#median for even number of data points
	if len(copy) % 2 == 0:
		index_of_median = int(len(copy) / 2)
		median = copy[index_of_median]
	#median for uneven number of data points
	else:
		index_of_medians = [int(len(copy) / 2), int((len(copy) / 2)+1)]
		median = (copy[index_of_medians[0]] + copy[index_of_medians[1]]) / 2

	#2nd and 3rd quartiles
	distance_btw_quartiles = int(len(copy)/4)

	second_quartile = copy[distance_btw_quartiles]
	third_quartile = copy[(distance_btw_quartiles*2)]

	return[typeval, mean, variance, minval, maxval, median, second_quartile, third_quartile]


"""
This function normalize the dataset to a value between 0 and 1. It takes the column, the minval and the maxval of the column
"""
def normalize(data, minval, maxval):
	normalized = []
	for elem in data: 
		r = (elem - minval) / (maxval - minval)
		normalized.append(r)
	return normalized

"""
This function takes the requested column /feature (except the first element which is the class label) and tests if the elements are floatable by calling the function is_number. 
If not, the index and the value is added to a list and this list is added to a list of outliers and empty element: outliers_empty.
If they are floatable they are floated
If age is below minaccepted or above maxacctepted index and value is appended to outliers_empty
The remaining values are appended to a list: accepted_inputs
From accepted_inputs random values are drawn and inserted at the indices of the empty or outlier data points
The updated list og all age answers is returned
"""
def ratio_cleaning(all_atributes, index_of_attribute, minaccepted, maxacctepted):
	#Finding outliers and empty fields
	outliers_empty = []
	answer = all_atributes[index_of_attribute][1:] #the desired column minus the first elem, which is the label
	temp = []
	accepted_inputs = []
	addedtogether = 0

	for i in range(len(answer)):
		#detecting non-floatable inputs
		if is_number(answer[i]) == False:
			temp.append(i)
			temp.append(answer[i])
		
			outliers_empty.append(temp)
			temp=[]
		
		if is_number(answer[i]) == True:
			answer[i] = float(answer[i])
			#detect outlier data
			if answer[i] >maxacctepted or answer[i] <minaccepted:

				temp.append(i)
				temp.append(answer[i])
			
				outliers_empty.append(temp)
				temp=[]
			else: 
				accepted_inputs.append(answer[i])
	
	#replacing outliers or empty cells with random numbers from the accepted inputs			
	for elem in outliers_empty:
		rand = choice(accepted_inputs)
		answer[elem[0]] = rand

	stringanswer = copy.copy(answer)
	for i in xrange(len(stringanswer)):
		stringanswer[i] = str(stringanswer[i])
	return stringanswer, answer


"""
This function expects a dataset and the index of the binary column to be cleaned.
If the lowercased answer contains "yes" or "1", the answer is 1.
If the lowercased answer contains "no" or "0", the anwer is 0.
Outliers or missing values are replaced by a random draw from the same distribution between yes and no as the correctly filled out answers. 
It returns the cleaned column in a 1D list
"""
def binary_cleaning(allattributes,index_of_attribute):
	attribute_val = all_atributes[index_of_attribute][1:] # the desired column minus the first elem, which is the label
	answer = np.copy(attribute_val)
	answer = answer.tolist()
	
	temp = []
	outliers_empty = []
	accepted_inputs = []
	for i in range(len(answer)): #we need the index to store with the outlier / empty
		
		if is_number(answer[i]) == False:
			answer[i] = str(answer[i])
			if 'yes' in answer[i].lower():
				answer[i] = "y"
				accepted_inputs.append(answer[1])
			elif answer[i] == '1':
				answer[i] = "y"
				accepted_inputs.append(answer[1])
			elif 'no' in answer[i].lower():
				answer[i] = "n"
				accepted_inputs.append(answer[1])
			elif answer[i] == '0':
				answer[i] = "n"
				accepted_input.append(answer[1])
			else: 
				temp.append(i)
				temp.append(answer[i])
				outliers_empty.append(temp)
				temp=[]
	yes = answer.count(1)
	no = answer.count(0)
	yesfrac = yes / (len(answer)- len(outliers_empty))
	nofrac = no / (len(answer)- len(outliers_empty))

	for elem in outliers_empty: #replacing outliers or empty cells with random numbers from the accepted inputs			
		rand = choice(accepted_inputs)
		answer[elem[0]] = rand
	return answer


"""
This function expects the cleaned column of a binary feature with letters "y" and "n". 
It returns a new feature vector with 0 and 1 instead
"""
def binary_cleaning_num(column):
	answer = np.copy(column)
	answer = answer.tolist()

	for i in range(len(answer)): #we need the index to store with the outlier / empty
		
		if answer[i] == "y":
			answer[i] = 1
		elif answer[i] == 'n':
			answer[i] = 0
	return answer


"""
This function cleans the operating system column. 
If it meets outlier data they are replaced by a random draw from either of the 3 classes. 
"""
def operating_sys_cleaning(dataset):
	attribute_val = all_atributes[3][1:] # the desired column minus the first elem, which is the label
	answer = np.copy(attribute_val)
	answer = answer.tolist()

	outliers_empty = []
	accepted_inputs = []

	for i in xrange(len(answer)):
		temp = []

		if "win" in answer[i].lower():
			answer[i] = "w"
		elif "lin" in answer[i].lower() or "ubun" in answer[i].lower():
			answer[i] = "l"
		elif "mac" in answer[i].lower() or "osx" in answer[i].lower():
			answer[i] = "m"
		else:
			temp.append(i) 
			temp.append(answer[i]) 
			outliers_empty.append(temp)

	for elem in outliers_empty: #replacing outliers or empty cells with random numbers from the accepted inputs			
		rand = choice(["w", "l", "m"])
		answer[elem[0]] = rand
	return answer

"""
This function expects the cleaned operating system column and binarizes the discrete values
"""
def operating_sys_cleaning_binary(column):
	answer = np.copy(column)
	answer = answer.tolist()
	for i in xrange(len(answer)):
		if answer[i] == "l":
			answer[i] = [0,1,0]
		elif answer[i] == "w":
			answer[i] = [1,0,0]
		elif answer[i] == "m":
			answer[i] = [0,0,1]
	return answer

#Standardizing for k-means

#Computing mean and variance
"""
This function takes a column and computes the mean and the variance the feature 
"""
def mean_variance(data):
	mean = sum(data) / len(data)
	
	#variance:
	su = 0	
	for elem in data:
		su += (elem - mean)**2
	variance = su/len(data)	
	return mean, variance


"""
This function calls mean_variance to get the mean and the variance for a feature
Then these values are used to normalize every datapoint to zero mean and unit variance.
A copy of the data is created. 
The normalized values are inserted at the old index in the copy 
The new, standardized data set is returned
"""
def meanfree(data):
	mean, variance = mean_variance(data)

	new = np.copy(data)
	for i in xrange(len(new)):
		#replacing at correct index in the copy
		new[i] = (new[i] - mean) / np.sqrt(variance)
	return new 

#########################################################################################
#
#										Calling
#
########################################################################################


all_atributes = make_lists_of_attributes_from_file(file2014)

#Programming skills
string_cleaned_prog_skills, cleaned_prog_skills = ratio_cleaning(all_atributes,1,0,10)
standardized_prog_skills = meanfree(cleaned_prog_skills)


#Tired of snow
snow_letter = binary_cleaning(all_atributes,8) #with letters for apriori
snow_num = binary_cleaning_num(snow_letter) # with binary vals for Kmeans and KNN

#Operating system
cleaned_os = operating_sys_cleaning(all_atributes) #For apriori
binary_os = operating_sys_cleaning_binary(cleaned_os) # for Kmeans and KNN 


#make arrays of the selected features with one person per array for apriori
apriori_features = []

for i in xrange(len(snow_letter)):
	temp = []
	temp.append(string_cleaned_prog_skills[i])
	temp.append(snow_letter[i])
	temp.append(cleaned_os[i])
	apriori_features.append(temp)


#Calling Apriori
prunedlen3 = Apriori.apriori(apriori_features)
print "*" *45
print "Apriori"
print "*" * 45
print "Longest frequent pattern:", prunedlen3

allpossible = Apriori.possible_comb(prunedlen3)

permutated = Apriori.permutations(allpossible)

Rules = Apriori.confidence(permutated, apriori_features)

print "Rules from longest frequent pattern and their confidence:", Rules

#make arrays of the selected features with one person per array for Kmeans and KNN
num_features = []

for i in xrange(len(snow_num)):
	temp = []
	temp.append(standardized_prog_skills[i])
	temp.append(binary_os[i][0]) #ugly hard-coded way of getting elements out of list
	temp.append(binary_os[i][1])
	temp.append(binary_os[i][2])
	temp.append(snow_num[i])
	num_features.append(temp)
num_features = np.array(num_features) #feature set [age, 3 binary values for operating sys, tiredness of snow]

#Calling k-means
Kmeans.kmeans(num_features)

#Calling KNN
#making a train and a test set. The label is the last value: tiredness of snow.
train = num_features[:50] #50 datapoints in train set
test = num_features[50:] # the remanining 17 datapoints in test set
k = 3
Error = KNN.eval(train, test, k)
print "*" *45
print "K-nearest neighbor"
print "*" * 45
print "k = ", k
print "Error on testset:", Error 







