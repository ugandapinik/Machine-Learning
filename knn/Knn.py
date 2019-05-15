import sys
import csv
import math
import operator

class Knn:

    def __init__(self, trainingSetFile, testingSetFile, k):
        self.trainingSet = []
        self.testingSet = []
        self.trainingSetFile = trainingSetFile
        self.testingSetFile = testingSetFile
        self.k = k

        self.handleDataSet()

    '''
    loading the dataset from file
    -----------------------------------------
    '''
    def handleDataSet(self):
        # Opening the training file
        with open(self.trainingSetFile, 'r') as csvTrainingFile:
            lines = csv.reader(csvTrainingFile, delimiter="\t")
            trainingDataset = list(lines)

            # Remove the first row as title from list
            trainingDataset.pop(0)

            for row in range(len(trainingDataset)):
                for column in range(9):
                    trainingDataset[row][column] = float(trainingDataset[row][column])

                self.trainingSet.append(trainingDataset[row])

        # Opening the training file
        with open(self.testingSetFile, 'r') as csvTestingFile:
            lines = csv.reader(csvTestingFile, delimiter="\t")
            testingDataset = list(lines)

            # Remove the first row as title from list
            testingDataset.pop(0)

            for row in range(len(testingDataset)):
                for column in range(9):
                    testingDataset[row][column] = float(testingDataset[row][column])

                self.testingSet.append(testingDataset[row])


    '''
    Evaluate the distance using the manhattan distance
    params: instance1   - get the full traningSet
    params: instance2   - get the full testingSet
    params: length      - get the length of the testInstance
    return: distance    - calculate the distance using Manhattan Distance
    '''
    def manhattanDistance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += abs(instance1[x] - instance2[x])

        return distance

    '''
    Evaluate the distance using the euclidean distance
    params: instance1   - get the full traningSet
    params: instance2   - get the full testingSet
    params: length      - get the length of the testInstance
    return: distance    - calculate the distance using Euclidean Distance
    '''
    def euclideanDistance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)

        # print(math.sqrt(distance))

        return math.sqrt(distance)


    '''
    This execute function run and find the predicted class
    and also Conditional Probability of the given testingSet
    '''
    def execute(self):
        predictions = []
        # print("For K: " + repr(k))
        # print("------------------------------------------------------")
        for x in range(len(self.testingSet)):
            neighbors = self.getNeighbors(self.trainingSet, self.testingSet[x], k)
            predicted = self.getResponse(neighbors)
            predictions.append(predicted)
            accuracy = self.getAccuracy(neighbors, predicted, k)


            # print("Predicted Class: " + predicted + " - Conditional Probability: " + "{0:.2f}".format(accuracy))
            print(predicted + "\t" + "{0:.2f}".format(accuracy))


    '''
    Find the nearest neighbors after calculating the
    distance using euclidean distance/manhattan distance
    ------------------------------------------------------------
    params: get the full traningSet
    params: get the full testingSet
    params: value of K
    return: nearest neighbors depending the value of K
    '''
    def getNeighbors(self, trainingSet, testInstance, k):
        distances = []
        length = len(testInstance)
        # print(length)

        for x in range(len(trainingSet)):
            dist = self.manhattanDistance(trainingSet[x], testInstance, length)
            distances.append((trainingSet[x], dist))

        distances.sort(key=operator.itemgetter(1))
        neighbors = []

        for x in range(k):
            neighbors.append(distances[x][0])

        return neighbors


    '''
    response function will find the predicted class
    depending on their closest neighbors
    '''
    def getResponse(self, neighbors):
        Votes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]

            if response in Votes:
                Votes[response] += 1
            else:
                Votes[response] = 1

        sortedVotes = sorted(Votes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]


    '''
    getting the conditional probability depending number of k
    and the predicted class we find. 
    this function will find how many times the predicted class appears
    neighbors and divide it with the number of K
    '''
    def getAccuracy(self, neighbors, predirected, k):
        # print(predictions[0])
        correct = 0
        # print(neighbors[0][-1])
        for x in range(len(neighbors)):
            if neighbors[x][-1] == predirected[0]:
                correct += 1

        # print(correct)
        return  (correct / k)







# getting trainingSetFileName
trainingSetFile = sys.argv[1]
# getting testingSetFileName
testingSetFile = sys.argv[2]
# getting number of K
# if k is not defined then k = 3 by default
k = int(sys.argv[3]) if len(sys.argv) > 3 else 3


knn = Knn(trainingSetFile, testingSetFile, k)
knn.execute()


