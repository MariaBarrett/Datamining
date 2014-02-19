from collections import Counter

transactions = [[ 1, 2, 3, 4, 5 ], [ 1, 3, 5 ], [ 2, 3, 5 ], [ 1, 5 ], [ 1, 3, 4 ], [ 2, 3, 5 ], [ 2, 3, 5 ], [ 3, 4, 5 ], [ 4, 5 ], [ 2 ], [ 2, 3 ], [ 2, 3, 4 ], [ 3, 4, 5 ]] 

minsupport = 0.5
minconfidence = 0.5

def firstlevelfreqitems(dataset):
	frequent_items={}
	for row in dataset:
		rowcount = Counter(set(row))
		print rowcount #accumulate? 
	for entry in rowcount:
		if value[entry] / len(dataset) > minsupport:
			frequent_items.insert(entry: value[entry] / len(dataset)])
	return frequent_items

firstlevelfreqitems(transactions)