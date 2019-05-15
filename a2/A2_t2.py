import csv
import sys
import numpy as np
import operator
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import log_loss

# KNN libraries
from sklearn.neighbors import KNeighborsClassifier
# import the metric model the check the accuracy
from sklearn import metrics

# import for generating confusion matrix
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from matplotlib import pyplot

# for roc curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score




'''
    LOAD THE DATASET
============================================================================'''
# loading the dataset
def loadDataSet(datasetFileName):
    # Opening the training file
    with open(datasetFileName, 'r') as datasetFile:
        lines = csv.reader(datasetFile, delimiter="\t")
        data = list(lines)
        return data




'''
    SPLIT A DATASET INTO (POSITIVE=1) / (NEGATIVE=0)
============================================================================'''
def train_test_split(dataset):
    zeroDataset = []
    oneDataset = []

    for row in range(len(dataset)):
        # dividing the list item into two different arraylist
        if dataset[row][-1] == '0':
            zeroDataset.append(dataset[row])

        elif dataset[row][-1] == '1':
            oneDataset.append(dataset[row])
    return zeroDataset, oneDataset


'''
    SELECTING THE FEATURES USING THE LOW VARIANCE
============================================================================'''
def feature_selection_with_low_var(dataset, variance=0.80):
    # using variance less than 0.8 we have to find
    # features using the sklearn feature_selection library
    selected = VarianceThreshold(variance * (1 - variance))

    data = selected.fit_transform(dataset)
    return data



'''
    SPLIT A DATASET INTO K FOLDS 
============================================================================'''
def cross_validation_split(zeroDataset, oneDataset, foldingNumber=10):

    folds = {}

    currentZeroIndex = 0
    currentOneIndex = 0
    zeroInc = round(len(zeroDataset) / foldingNumber)
    zeroIndex = round(len(zeroDataset) / foldingNumber)
    oneInc = round(len(oneDataset) / foldingNumber)
    oneIndex = round(len(oneDataset) / foldingNumber)


    for item in range(foldingNumber): # number of fold is 10
        #divide the training and testing dataset according the folding
        if (item == foldingNumber - 1):
            tempZeroDataset = zeroDataset[currentZeroIndex:]
            tempOneDataset = oneDataset[currentOneIndex:]
        else:
            tempZeroDataset = zeroDataset[currentZeroIndex:zeroIndex]
            tempOneDataset = oneDataset[currentOneIndex:oneIndex]

        folds["folds_{0}".format(item)] = tempZeroDataset + tempOneDataset


        # increasing the value
        currentZeroIndex += zeroInc
        currentOneIndex += oneInc
        zeroIndex += zeroInc
        oneIndex += oneInc


    # for i in range(foldingNumber):
    #     print(len(folds["folds_{0}".format(i)]))
    return folds




'''
    TESTING AND TRAINNING DATA CLASS LABEL
    :param foldings -  are the 10 foldings
    :param label - which folding choose as testing dataset, default = 0
============================================================================'''
def find_class_label(folds, type = ""):
    attributes = []
    label = []

    for item in folds:
        attributes.append(item[:-1])
        label.append(item[-1])

    attributes = list(attributes)
    label = list(label)

    if type == "attributes":
        return attributes
    elif type == "label":
        return label
    else:
        return attributes, label


'''
    SPLITTING DATA INTO TRAINING AND TESTING DATA
    :param folding - list of folding
    :param label - every single folding level 
                    
============================================================================'''
def splitting_train_test_data(foldings, label = 0):

    X_test, y_test = find_class_label(foldings["folds_{0}".format(label)])
    # remove the item from the dictionary


    X_train = []
    y_train = []

    for item in foldings:
        if(item != "folds_{0}".format(label)):
            #print(item)
            X_train += find_class_label(foldings[item], "attributes")
            y_train += find_class_label(foldings[item], "label")


    return X_train, y_train, X_test, y_test



'''
    KNN CLASSIFIERS
    :param folds - total folds
    :param - classno = class label no - default = 2
    :param - knn_range = range of the knn - default = 10
============================================================================'''

def knn_classifiers(folds,  classno = 2, knn_range = 10):
    scores = {}
    scores_list = []
    average_score = {}

    for k in range(1, knn_range):
        if k % 2 == 1:
            #print(k)
            for label in range(len(folds)):
                X_train, y_train, X_test, y_test = splitting_train_test_data(folds, label)
                #print(len(X_train))
                # selecting the featuresX_test
                # ---------------------------------------------------------------------------
                X_train = feature_selection_with_low_var(X_train).astype(np.float64)
                X_test = feature_selection_with_low_var(X_test).astype(np.float64)

                knn = KNeighborsClassifier(n_neighbors=k, p=classno)
                knn.fit(X_train, y_train)
                predicted = knn.predict(X_test)
                #print(metrics.accuracy_score(y_test, predicted))


                scores[k] = metrics.accuracy_score(y_test, y_pred=predicted)
                scores_list.append(metrics.accuracy_score(y_test, y_pred=predicted))

            average_score[k] = format(np.mean(scores_list), '.2f')

    #display_k_graph(average_score)



    best_k = max(average_score.items(), key=operator.itemgetter(1))[0]
    average = max(average_score.items(), key=operator.itemgetter(1))[1]
    return best_k, average






'''
    KNN CLASSIFIER
    :param folds - total folds
    :param kvalue - best k value
============================================================================'''
def knn_classifier(folds, kvalue):
    scores = []
    log_v = []
    pred_values = []
    true_values = []
    scores_list = []

    counter = 1
    for label in range(len(folds)):
        X_train, y_train, X_test, y_test = splitting_train_test_data(folds, label)
        # print(len(X_train))
        # selecting the featuresX_test
        # ---------------------------------------------------------------------------
        X_train = feature_selection_with_low_var(X_train).astype(np.float64)
        X_test = feature_selection_with_low_var(X_test).astype(np.float64)

        knn = KNeighborsClassifier(n_neighbors=kvalue, p=2)
        knn.fit(X_train, y_train)
        predicted = knn.predict(X_test)

        probs = knn.predict_proba(X_test)
        log_value = log_loss(find_class_label(folds["folds_{0}".format(label)], type="label"), probs)
        log_v.append(log_value)

        scores.extend(probs[:, 1])
        pred_values.extend(predicted)
        true_values.extend(find_class_label(folds["folds_{0}".format(label)], type="label"))



        #print(metrics.accuracy_score(y_test, predicted))
        scores[counter] = metrics.accuracy_score(y_test, y_pred=predicted)
        scores_list.append(metrics.accuracy_score(y_test, y_pred=predicted, normalize=False))
        counter += 1

        # print(repr(k) + "-" + "{0:.2f}".format(np.mean(scores_list)))

    log_value_average = format(np.mean(log_v), '.2f')

    for i in range(len(pred_values)):
        scores[i] = scores[i]
        pred_values[i] = int(pred_values[i])
        true_values[i] = int(true_values[i])




    return pred_values, true_values, scores, log_value_average



'''
    DISPLAY ROC CURVE
    :param - actual value
    :param - predicted value
============================================================================'''
def display_roc_graph(true_values, pred_values):

    auc = roc_auc_score(true_values, pred_values)
    print('AUC: %.3f' % auc)

    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(true_values, scores)

    # Plot ROC curve
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    pyplot.ylabel('TPR')
    pyplot.xlabel('FPR')
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='.')
    # show the plot
    pyplot.show()



'''
    DISPLAY ALL K VALUE GRAPH
============================================================================'''
def display_k_graph(kandscore):

    k_range = []
    scores = []

    for item in kandscore:
        k_range.append(item)
        scores.append(kandscore[item])

    pyplot.plot(k_range, scores)
    pyplot.xlabel('Value of K for KNN')
    pyplot.ylabel('Testing Accuracy')
    pyplot.show()




'''
    MAIN FUNCTION
============================================================================'''
if __name__ == "__main__":

    # get the filename from username
    filename = sys.argv[1]

    # load the dataset
    # ---------------------------------------------------------------------------
    dataset = loadDataSet(filename)

    # splitting data depending on the data depending on the class
    # for the class 1 (yes) will be in the oneDataset
    # fo the class 0 (no) will be in the zeroDataset
    #---------------------------------------------------------------------------
    zeroDataset, oneDataset = train_test_split(dataset)

    # getting the folds
    #---------------------------------------------------------------------------
    folds = cross_validation_split(zeroDataset, oneDataset, foldingNumber = 10)

    # getting the training and testing data
    #---------------------------------------------------------------------------

    best_k, accuracy = knn_classifiers(folds, classno=2, knn_range=35)

    #---------------------------------------------------------------------------
    print("K: " + repr(best_k) + " Accuracy: " + accuracy)
    print("----------------------------------------------------------------------")
    pred_values, true_values, scores, log_value_average = knn_classifier(folds, best_k)
    # con_mat = confusion_matrix(y_train, pred_values, [0, 1])
    cofusion_matrix = confusion_matrix(true_values, pred_values)



    y_actu = pd.Series(true_values, name='actual')
    y_pred = pd.Series(pred_values, name='predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)

    # print log loss
    print("Loss Function: " + repr(log_value_average))

    print("Confusion Matrix: ")
    print("----------------------------------------------------------------------")
    print(df_confusion)

    print("Classification Report: ")
    print("----------------------------------------------------------------------")
    print(classification_report(true_values, pred_values))

    # Display the roc curve
    #---------------------------------------------------------------------------
    display_roc_graph(true_values, pred_values)










