from __future__ import division
from collections import Counter
import itertools
import numpy as np

transactions = [[ 1, 2, 3, 4, 5 ], [ 1, 3, 5 ], [ 2, 3, 5 ], [ 1, 5 ], [ 1, 3, 4 ], [ 2, 3, 5 ], [ 2, 3, 5 ], [ 3, 4, 5 ], [ 4, 5 ], [ 2 ], [ 2, 3 ], [ 2, 3, 4 ], [ 3, 4, 5 ]] 

minsupport = 0.30
minconfidence = 0.4
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
	joinedsets = {}
	temp = {}
	
	for a in itertools.combinations(frequent_items, 2): #using a library function to generate combinations of frequent items
		item1 = a[0]
		item2 = a[1]

		for row in dataset:
			#count joint probabilities and store count in a dictionary	
			if item1 in set(row) and item2 in set(row):
				if str(item1)+str(item2) in temp: 
					temp[str(item1)+str(item2)] += 1
				else:
					temp[str(item1)+str(item2)] = 1

	#Only saving in joinedsets if above threshold.
	for entry in temp: 
		if temp[entry] /len(dataset) > minsupport:
			joinedsets[entry] = temp[entry]
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
	C2combis = [int(float(num)) for num in len2.keys()] #turning list of strings into list of int
	newC2 = []

	#creating possible combinatioons of frequent combinations from frequent sets
	for combi in C2combis:
		newC2.append(map(int, str(combi))) #making a list of list of item sets 

	#C3
	len3 = joinSets_afterC2(newC2, transactions)
	pruned_len3 = prune(len3, newC2)

	#c4 
	len4 = joinSets_afterC2(len3, transactions)
	pruned_len4 = prune(len4, len3) #Nothing returned

	return pruned_len3

prunedlen3 = apriori(transactions)


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
"""

"""
def confidence(all_possible_from_maxlen):
	allpermutations = []

	for comb in all_possible_from_maxlen:
		temp = []
		for c in itertools.permutations(comb,2):
			temp.append(c)
		allpermutations.append(temp)
	print len(allpermutations[0])
	for x in allpermutations:
		for pair in x:
			print pair
			print pair[0]
			print pair[1]
			if pair[0] in pair[1]:
				del(pair)
	print len(allpermutations[0])


allpossible = possible_comb(prunedlen3)
confidence(allpossible)