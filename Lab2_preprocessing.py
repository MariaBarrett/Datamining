from __future__ import division
import numpy as np
import csv
from random import choice
from collections import Counter

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
This function takes the first column (except the first element which is the class label) and tests if the elements are floatable by calling the function is_number. 
If not, the index and the value is added to a list and this list is added to a list of outliers and empty element: outliers_empty.
If they are floatable they are floated
If age is below 17 or above 110 index and value is appended to outliers_empty
The remaining values are appended to a list: accepted_inputs
From accepted_inputs random values are drawn and inserted at the indices of the empty or outlier data points
The updated list og all age answers is returned
"""
def age_cleaning(all_atributes):

	#Finding outliers and empty fields
	outliers_empty = []
	age_answer = all_atributes[0][1:] #the first column minus the first elem, which is the label
	temp = []
	accepted_inputs = []
	addedtogether = 0

	for i in range(len(age_answer)):
		#detecting non-floatable inputs
		if is_number(age_answer[i]) == False:
			temp.append(i)
			temp.append(age_answer[i])
		
			outliers_empty.append(temp)
			temp=[]
		
		if is_number(age_answer[i]) == True:
			age_answer[i] = float(age_answer[i])
			#detect outlier data
			if age_answer[i] >110 or age_answer[i] <17:

				temp.append(i)
				temp.append(age_answer[i])
			
				outliers_empty.append(temp)
				temp=[]
			else: 
				accepted_inputs.append(age_answer[i])
	#replacing outliers or empty cells with random numbers from the accepted inputs			
	for elem in outliers_empty:
		rand = choice(accepted_inputs)
		age_answer[elem[0]] = rand

	return age_answer

def describing_age(cleaned_data):
	accummulated = 0
	#mode
	counted = Counter(cleaned_age)
	typevaltuple = counted.most_common(1)
	typeval = typevaltuple[0][0]

	#max, min
	cleaned_data.sort()
	print cleaned_age
	maxval = (cleaned_age[-1])
	minval = (cleaned_age[0])

	#median
	index_of_median = int(len(cleaned_age) / 2)
	median = cleaned_age[index_of_median]

	#quartiles
	distance_btw_quartiles = int(len(cleaned_data)/4)
	first_quartile = cleaned_age[0::distance_btw_quartiles]
	second_quartile = cleaned_age[distance_btw_quartiles::(2*distance_btw_quartiles)]
	third_quartile = cleaned_age[(distance_btw_quartiles*2)::distance_btw_quartiles*3]
	fourth_quartile = cleaned_age[(distance_btw_quartiles*3)::-1]

	for elem in cleaned_age:
		accummulated += elem
	mean = accummulated / len(cleaned_age)
	return typeval, maxval, minval, median, mean

def normalizing(cleaned_data):
	typeval, maxval, minval, median, mean = describing_age(cleaned_data)


all_atributes = make_lists_of_attributes_from_file(file2014)
cleaned_age = age_cleaning(all_atributes)
describing_age(cleaned_age)