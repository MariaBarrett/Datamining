from __future__ import division
import numpy as np
import csv
from random import choice
from collections import Counter

"""
README
File is converted to UTF-8 before opening them using code
"""
#Opening file, stripping and splitting. The structure is as follows [header[first header][second header]...][second person[first answer]...]
#Attributes variables

file2013 = "Data_Mining_Student_DataSet_Spring_2013_Fixed.csv"
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

"""
This function tests if a string is a number. I didn't write it myself. It's from http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-in-python
"""
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

"""
This function takes the requested column /feature (except the first element which is the class label) and tests if the elements are floatable by calling the function is_number. 
If not, the index and the value is added to a list and this list is added to a list of outliers and empty element: outliers_empty.
If they are floatable they are floated
If age is below minaccepted or above maxacctepted index and value is appended to outliers_empty
The remaining values are appended to a list: accepted_inputs
From accepted_inputs random values are drawn and inserted at the indices of the empty or outlier data points
The updated list og all age answers is returned
"""
def ratio_cleaning(all_atributes, index_of_attribute,minaccepted,maxacctepted):
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
	print outliers_empty
	print accepted_inputs
	#replacing outliers or empty cells with random numbers from the accepted inputs			
	for elem in outliers_empty:
		rand = choice(accepted_inputs)
		answer[elem[0]] = rand

	return answer

def describing_age(cleaned_data):
	"""
	This function takes the cleaned data and describes through 
	mode, mean, variance, max, min, median and 2nd and 3rd quartiles
	"""
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

def normalize(data, minval, maxval):
	normalized = []
	for elem in data: 
		r = (elem - minval) / (maxval - minval)
		normalized.append(r)
	return normalized

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
				answer[i] = 1
				accepted_inputs.append(answer[1])
			elif answer[i] == '1':
				answer[i] = 1
				accepted_inputs.append(answer[1])
			elif 'no' in answer[i].lower():
				answer[i] = 0
				accepted_inputs.append(answer[1])
			elif answer[i] == '0':
				answer[i] = 0
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
		
#Calling

all_atributes = make_lists_of_attributes_from_file(file2014)

cleaned_age = ratio_cleaning(all_atributes,0,17,110)
cleaned_prog_skills = ratio_cleaning(all_atributes,1,0,10)

desc_age = describing_age(cleaned_age)


normdata = normalize(cleaned_age, desc_age[3], desc_age[4])

Snow = binary_cleaning(all_atributes,8)

"""
Todo
normalize
z-score
describe data after normalizing
"""


