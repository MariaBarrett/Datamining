from __future__ import division
import numpy as np
import csv
from random import choice
from collections import Counter
import itertools

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

	return answer


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
		rand = choice([0.0, 1.0, 2.0])
		answer[elem[0]] = rand
	return answer


#############################################################################################
#
#								Apriori
#
#############################################################################################

minsupport = 0.1
minconfidence = 0.5
"""
This function takes a dataset and finds frequent items. It returns a dictionary of items 
which are frequent enough to pass the support threshold. 
"""
def countSupport(dataset):
	itemcount = {}
	frequent_items={} # Contains item: support 
	for row in dataset:
		for val in set(row):
			if val in itemcount:
				itemcount[val] = itemcount[val] +1
			else:
				itemcount[val] = 1 

	for entry in itemcount:
		if itemcount[entry] / len(dataset) > minsupport:
			frequent_items[entry] = itemcount[entry] 	
	return frequent_items
	
"""
This function expects a dict of frequent items and a dataset. It uses a libray function to generate permutations of items in the list.
It counts the joint appearances of item 1 and 2. It uses the set of each row to avoid double count in the case where an item appears two time in the same row. 
It divides the joint probability by count of item 1 to get the support and returns a new dictioary, joinedsets, if result is above threshold. 
"""

def joinSets_C2(frequent_items, dataset):
	joinedsets = []

	
	for a in itertools.combinations(frequent_items, 2): #using a library function to generate combinations of frequent items
		temp = []
		item1 = a[0]
		item2 = a[1]
		count = 0
		for row in dataset:
			#count joint probabilities and store count in a dictionary	
			if item1 in set(row) and item2 in set(row):
				count +=1
		#only saving if avobe threshold
		if count / len(dataset) > minsupport:

			temp.append(item1)
			temp.append(item2)
			joinedsets.append(temp)

	return joinedsets

"""
This function makes new permutations by combining frequent itemsets. 
Only itemsets that have k-1 items in common are combined. 
If the new combination is also a frequent combination 
"""
def joinSets_afterC2(frequentcombi, dataset):
	frequent_items_dict = {}

	#making a set of all items:
	setofitem = []
	[setofitem.append(num) for elem in frequentcombi for num in elem]
	setofitem = set(setofitem)

	possible = []
	for frqset in frequentcombi:
		for item in setofitem:
			if item not in frqset:
				newtemp =[]
				for elem in frqset:
					newtemp.append(elem) 
				newtemp.append(item)#append single value to all lists if not already there:
				possible.append(newtemp)
	return possible

"""
This function prunes by making all possible combinations that are shorter than k-1.
It stores all possible combinations in a temporary list and tries them on the 
previous list of frequent sets. Only if all combinations are in this list, the
candidate is stored in a new list
"""
def prune(possible_candidates, previous_freq):
	#turning concatenated itemnames into lists of items
	approved = []
	
	for i in xrange(len(possible_candidates)):
		temp =[] #making a list of all possible combinations of lenght 2
		for comb in itertools.combinations(possible_candidates[i],2):
			temp.append(list(comb))
			count = 0
		
		maxpoint = len(temp) #max combination that can be matched
		for c in temp:
			if c in previous_freq:
				count += 1
		if count == 3: #if all combinations are in the previous list of frequent sets
			approved.append(possible_candidates[i])
	return approved
		
"""
Here everything is called. lenght is increased manually if there are still pruned examples in the previous
Len4 is the first that does not return anything
"""
def apriori(dataset):
	#C1
	freq_items = countSupport(dataset) 

	#C2
	len2 = joinSets_C2(freq_items, dataset) 

	#C3
	len3 = joinSets_afterC2(len2, selected_features)

	pruned_len3 = prune(len3, len2)

	return pruned_len3



"""
This function tries to make possible combinations of the max lenght pruned sets.
It expects the pruned max lenght sets. It expects the following format[[list of set 1][list of set 2]...]
It returns a list per input combinations. Each list contains a tuple of up to one element less that the lenght of the input. 
"""
def possible_comb(pruned_maxlenght):
	possible = []
	for comb in pruned_maxlenght:
		temp = []
		for i in xrange(len(comb)):
			for c in itertools.combinations(comb,i):
				if c != ():
					temp.append(c)
		possible.append(temp)
	return possible


def permutations(all_possible_from_maxlen):
	allpermutations = []

	for comb in all_possible_from_maxlen:
		temp = []
		for c in itertools.permutations(comb,2):
			temp.append(c)
		allpermutations.append(temp)

	#delete permutations if one value is contained in the other (not perfect - works only for first digit)
	for x in allpermutations:
		for i,n in enumerate(x):
			if n[0][0] in n[1] or n[1][0] in n[0]:
				del x[i]
			if len(n[0]) > 1:
				if n[0][1] in n[1]:
					del x[i]
			if len(n[1]) > 1:
				if n[1][1] in n[0]:
					del x[i]

	return allpermutations #this is a list of permutations ready for calculating confidence


"""
This function expects the permutated pairs and counts the joint probability of the pairs and the occurence of the first part of pair
If the confidence is above threshold, the pair and the confidence is stored in a dictionary, which is returned. 
"""
def confidence(permutated, dataset):
	confi = {}
	for sublist in permutated:
		for pair in sublist:
			count1 = 0
			count2 = 0
			list(pair)
			for row in dataset:
				if set(pair[0]).issubset(row):
					count1 +=1 
					if set(pair[0]).issubset(row) and set(pair[1]).issubset(row):
						count2 +=1
			confidence = count2 / count1
			if confidence > minconfidence:
				confi[(str(pair[0])+str(pair[1]))] = confidence
	return confi
#########################################################################################
#
#										Calling
#
########################################################################################


all_atributes = make_lists_of_attributes_from_file(file2014)

#Programming skills
cleaned_prog_skills = ratio_cleaning(all_atributes,1,0,10)
desc_prog_skills = describing(cleaned_prog_skills)
norm_prog_skills = normalize(cleaned_prog_skills, desc_prog_skills[3], desc_prog_skills[4])

#Tired of snow
snow = binary_cleaning(all_atributes,8)

#Operating system
cleaned_os = operating_sys_cleaning(all_atributes)

#make arrays of the selected features with one person per array
selected_features = []

for i in xrange(len(snow)):
	temp = []
	temp.append(norm_prog_skills[i])
	temp.append(snow[i])
	temp.append(cleaned_os[i])
	selected_features.append(temp)


#Calling Apriori
prunedlen3 = apriori(selected_features)

print prunedlen3

allpossible = possible_comb(prunedlen3)

permutated = permutations(allpossible)

frequentpatterns = confidence(permutated, selected_features)

print frequentpatterns



