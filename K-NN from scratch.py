import math
import operator
import numpy
from sklearn.model_selection import StratifiedKFold, KFold
from decimal import Decimal
from sklearn.preprocessing import MinMaxScaler

def loadDataset(filename, k, datafile):
    #Load selected datafile
    with open(filename, 'rt'):
        hasil, pow_value, hasil1 = 0, 0, 0
        dataset = numpy.loadtxt(filename, delimiter=",")
        lines = [0] * len(dataset)
        if datafile == 1:
            #Split the Pima Indians Diabetes dataset using Stratified Kfold, n-fold = 10
            kfold = StratifiedKFold(10, False, 0).split(dataset, lines)
        else:
            #Split the Housing Data dataset using Kfold, n-fold = 10
            kfold = KFold(10, True, 1).split(dataset)
            
        dataset2 = dataset
        
        #MinMaxScaler is used for feature scaling, in range 0 - 1
        scaler = MinMaxScaler().fit(dataset)
        dataset = scaler.fit_transform(dataset)
        
        #Restore only the class attribute to the previous value before scaling
        for i in range(len(dataset)):
            if datafile == 1:
                dataset[i][8] = dataset2[i][8]
            elif datafile == 2:
                dataset[i][13] = dataset2[i][13]
        
        print('1. Cosine Similarity\n2. Euclidian Distance\n3. Manhattan Distance\n4. Minkowski Distance')
        algo_dist = int(input('Choose a distance algorithm : '))
       
        #If Minkowski algorithm is selected then we have to enter the pow value
        if algo_dist == 4:
            pow_value = int(input('Enter the pow value : ' ))
            
        for train, test in kfold:
            #print('Data train: %s, \nData test: %s' % (dataset[train], dataset[test]))
            predictions = []
            for x in range(len(dataset[test])):
                neighbors = getNeighbors(dataset[train], dataset[test][x], k, pow_value, algo_dist)
                result = getResponse(neighbors, datafile, k)
                predictions.append(result)
                #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1])) 
            if datafile == 1:
                accuracy = getAccuracy(dataset[test], predictions)
                hasil = hasil + accuracy
            else:
                accuracy = getMAPE(dataset[test], predictions)
                accuracy1 = getRMSE(dataset[test], predictions)
                hasil = hasil + accuracy
                hasil1 = hasil1 + accuracy1
            #print('Akurasi ' + repr(i) + ' : ' +repr(accuracy) + '%')
            
        #hasil is the average accuracy because we use 10-fold cross-validation
        if datafile == 1:
            print('Accuracy : ' + repr(hasil/10) + '%')
        else:
            print('MAPE : ' + repr(hasil/10))
            print('RMSE : ' + repr(hasil1/10))

def cosineSimilarity(vector_1, vector_2, length):
    sumv1, sumv2, sumv1v2 = 0, 0, 0
    for i in range(length):
        x = vector_1[i]
        y = vector_2[i]
        sumv1 += x * x
        sumv2 += y * y
        sumv1v2 += x * y
    return -(sumv1v2 / (math.sqrt(sumv1) * math.sqrt(sumv2)))

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance = distance + pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)

def manhattanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance = distance + abs(instance1[x]-instance2[x])
    return distance

def nth_root(value, n_root):
    #Calculate nth root for Minkowski algorithm
    root_value = 1/float(n_root)
    return round (Decimal(value) ** Decimal(root_value),3)

def minkowskiDistance(instance1, instance2, length, pow_value):
    distance = 0
    for x in range(length):
        distance = distance + pow(abs(instance1[x]-instance2[x]), pow_value)
    return nth_root(distance, pow_value)

def getNeighbors(trainingSet, testInstance, k, pow_value, algo_dist):
    #Returns k most similar neighbors from the training set for a given test instance using selected distance algorithm
    distances = []
    length = len(testInstance)-1
    
    for x in range(len(trainingSet)):
        if algo_dist == 1:
            dist = cosineSimilarity(testInstance, trainingSet[x], length)
        elif algo_dist == 2:
            dist = euclideanDistance(testInstance, trainingSet[x], length)
        elif algo_dist == 3:
            dist = manhattanDistance(testInstance, trainingSet[x], length)
        elif algo_dist == 4:
            dist = minkowskiDistance(testInstance, trainingSet[x], length, pow_value)
            
        distances.append((trainingSet[x], dist))
        
    #Sort the distance in ascending order
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    
    for i in range(k):
        #Get K nearest neighbors from previous sorted list
        neighbors.append(distances[i][0])
    return neighbors

def getResponse(neighbors, choose, k):
    #Get the majority voted response from a number of neighbors.
    #Response is the class attribute
    if choose == 1:
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1] 
            if response in classVotes:
                classVotes[response] = classVotes[response] + 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]
    elif choose == 2:
        sum = 0
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            sum = sum + response
        return (sum/k)

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) *100.0
    
def getMAPE(testSet, predictions):
    #Calculate Mean Absolute Percentage Error (MAPE) to evaluate the K-NN algorithm for regression
    correct = 0
    for x in range(len(testSet)):
        correct += abs((testSet[x][13] - predictions[x]) / testSet[x][13])
    return (correct/float(len(testSet))) * 100

def getRMSE(testSet, predictions):
    #Calculate Root Mean Square Error (RMSE) to evaluate the K-NN algorithm for regression 
    correct = 0
    for x in range(len(testSet)):
        correct += pow((testSet[x][13] - predictions[x]), 2) / float(len(testSet))
    return (pow(correct, 0.5))

def main():
    print('1. Pima Indians Diabetes\n2. Housing Data')
    datafile = int(input('Choose a dataset : '))
    if datafile == 1:
        file = 'pima-indians-diabetes1.csv' #this dataset is used to implement K-NN for classification
    elif datafile == 2:
        file = 'housingdata.csv' #this dataset is used to implement K-NN for regression
    k = int(input('Enter k : ' ))
    loadDataset(file, k, datafile)

main()