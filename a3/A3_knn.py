import pandas as pd
import csv

from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import matplotlib as plt

'''
    LOAD THE DATASET 
    :param - getting the dataset filename
============================================================================'''
# loading the dataset
def loadDataSet(datasetFileName):
    # Opening the training file
    with open(datasetFileName, 'r') as datasetFile:
        lines = csv.reader(datasetFile, delimiter="\t")
        data = list(lines)
        result = pd.DataFrame(data)
        return result


def feature_selection_with_low_var(dataset, variance=0.80):
    # using variance less than 0.8 we have to find
    # features using the sklearn feature_selection library
    selected = VarianceThreshold(variance * (1 - variance))

    data = selected.fit_transform(dataset)
    return data




dataframe = loadDataSet("trainingdataset.tsv")
print(dataframe.shape)



'''
    SPLITTING DATASET
============================================================================'''
X = dataframe.iloc[:,:71]
y = dataframe.iloc[:,-1]

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# loading library
from sklearn.neighbors import KNeighborsClassifier

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, pred))


k_range = list(range(1,30))

# subsetting just the odd ones
neighbors = []


# empty list that will hold cv scores
cv_scores = []
scores = {}

# perform 10-fold cross validation
for k in k_range:
    if k % 2 == 1:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
        neighbors.append(k)

print(cv_scores)

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)


'''
    FOLDING DATASET
============================================================================'''
# kfold = KFold(10, True, 1)
# # enumerate splits
# for train, test in kfold.split(dataframe):
# 	print(train)


#
# from sklearn.model_selection import train_test_split, KFold
#
# #split dataset into train and test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
#
