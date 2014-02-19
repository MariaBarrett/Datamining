from string import split
import math
import operator
 
def majorityCnt(classlist):
    classcount={}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote]=0
        classcount[vote] += 1
    sortedClassCount=sorted(classcount.iteritems(),key=operator.itemgetter(1),reverse=True)
    print sortedClassCount
    return sortedClassCount[0][0]
 
def entropy(dataset):
    n=len(dataset)
    labels={}
    for record in dataset:
        label=record[0]
        if label not in labels.keys():
            labels[label]=0
        labels[label]+=1
    ent=0.0
    for key in labels.keys():
        prob=float(labels[key])/n
        ent= -prob*math.log(prob,2)
    return ent
 
def splitDataset(dataset,nclom,value):
    retDataSet=[]
    for record in dataset:
        if record[nclom] == value:
            reducedRecord=record[:nclom]
            reducedRecord.extend(record[nclom+1:])
            retDataSet.append(reducedRecord)
    return retDataSet
 
def chooseBestFeatureToSplit(dataset):
    numberFeature=len(dataset[0])-1
    baseEntropy=entropy(dataset)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numberFeature):
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
 
def buildTree(dataset,labels):
    classlist=[ x[-1] for x in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(classlist)==1:
        return majorityCnt(classlist)
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
 
def classify(tree,labels,testvec):
    firstStr = tree.keys()[0]
    secondDict = tree[firstStr]
    featIndex = labels.index(firstStr)
    for key in secondDict.keys():
        if testvec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],labels,testvec)
            else: classLabel = secondDict[key]
    try:
        return classLabel
    except:
        return 1
 

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
    return trainset, testset

#Calling function to create train and test set
trainset, testset = file_to_train_and_test(mainfile)


 
nfeature=len(trainset[0])
labels=["att"+str(i) for i in xrange(1,nfeature)] #not taking the class
labels2=[x for x in labels]
tree=buildTree(trainset, labels)
#print tree#['att4']
 
nPos=0
for r in testset:
    ret=classify(tree, labels2, r)
    if ret==r[0]:
        nPos +=1
ntest=len(testset)
print "The pass rate is " + str(nPos/float(ntest))