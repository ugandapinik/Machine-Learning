import csv
import sys



def loadingDataSet(fileName):
    # Opening the training file
    with open(fileName, 'r') as confusionMatrixFile:
        # removing the header with next function
        next(confusionMatrixFile)

        # removing the first column with strip
        matrix = [x.strip().split()[1:] for x in confusionMatrixFile]
        matrix = list(matrix)

        for row in range(len(matrix)):
            for column in range(len(matrix)):
                matrix[row][column] = int(matrix[row][column])

            confusionMatrix.append(matrix[row])



def accuracy(confusionMatrix):
    diagonalSum = 0
    totalSum = 0

    for row in range(len(confusionMatrix)):
        # get the diagonal sum of the matrix
        diagonalSum += confusionMatrix[row][row]
        for column in range(len(confusionMatrix)):
            # get the total sum of the matrix
            totalSum += confusionMatrix[row][column]

    accuracy = diagonalSum / totalSum
    return accuracy




def precision(label, confusionMatrix):
    truePositive = confusionMatrix[label]
    predictedPostiveSum = 0
    for item in range(len(confusionMatrix)):
        predictedPostiveSum += confusionMatrix[item]

    precision = truePositive / predictedPostiveSum
    return precision




def recall(label, confusionMatrix):

    truePositive = confusionMatrix[label][label]
    #print(truePositive)
    realPositiveSum = 0
    for row in range(len(confusionMatrix[label])):
        realPositiveSum += confusionMatrix[row][label]

    recall = truePositive / realPositiveSum
    return recall



def specificity(label, confusionMatrix):
    sumRealNegative = 0
    trueNegative = 0
    for row in range(len(confusionMatrix)):
        for column in range(len(confusionMatrix)):
            if row != label and column != label:
                trueNegative += confusionMatrix[row][column]

        if row != label:
            sumRealNegative += confusionMatrix[label][row]

    sumRealNegative = trueNegative + sumRealNegative
    specificity = trueNegative / sumRealNegative

    return specificity



def FDR(label, confusionMatrix):
    falsePositive = 0
    positive = 0
    for row in range(len(confusionMatrix)):
        positive += confusionMatrix[row]
        if row != label:
            falsePositive += confusionMatrix[row]


    FDR = falsePositive / positive
    return FDR



if __name__ == '__main__':
    #load the file with the system parameters

    filename = sys.argv[1]
    confusionMatrix = []
    loadingDataSet(filename)
    #print(confusionMatrix)

    # printing the accuracy
    print("Ac:" + "\t{0:.2f}".format(accuracy(confusionMatrix)))

    counter = 1
    print("\tP" + "\tR" + "\tSp" + "\tFDR")
    for row in range(len(confusionMatrix)):
        # print("C" + repr(counter) + "\t{0:.2f}".format(precision(row, confusionMatrix[row])))
        print("C" + repr(counter)
              + "\t{0:.2f}".format(precision(row, confusionMatrix[row]))
              + "\t{0:.2f}".format(recall(row, confusionMatrix))
              + "\t{0:.2f}".format(specificity(row, confusionMatrix))
              + "\t{0:.2f}".format(FDR(row, confusionMatrix[row])))
        counter += 1