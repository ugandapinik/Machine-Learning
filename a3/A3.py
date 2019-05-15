# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.metrics import classification_report, precision_recall_curve, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


'''
    LOAD THE DATASET 
    :param - data filename
============================================================================'''
# loading the dataset
def loadDataSet(datasetFileName):
    from numpy import genfromtxt
    sample = pd.read_csv(datasetFileName, sep='\t',header=None, dtype=None)
    # sample = sample[:20]
    #print(sample.columns)
    print("dimension of diabetes data: {}".format(sample.shape))
    # descriptions
    # class distribution
    return sample








def GridSearchCVforRandomForest(train_set, test_set):
    dataFrame = loadDataSet(train_set)
    dataFrameTest = loadDataSet(test_set)

    # print(dataFrame)
    X_Train = dataFrame.iloc[:, :71]
    Y_Train = dataFrame.iloc[:, -1]
    X_Yest = dataFrameTest.iloc[:, :71]

    # sm = SMOTE(random_state=42)
    # X_res, y_res = sm.fit_resample(X, y)
    # print('Resampled dataset shape %s' % Counter(y_res))
    '''priting group of number of 0 and one // 0 = 3469 and 1 = 348'''
    print(dataFrame.groupby(Y_Train).size())


    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    #X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.20, random_state=99, stratify=y)

    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf = RandomForestClassifier(n_estimators=100,
                                 criterion='gini',
                                 min_samples_split=8,
                                 min_samples_leaf=3,
                                 min_weight_fraction_leaf=0.0,
                                 max_features='auto',
                                 max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,
                                 min_impurity_split=None,
                                 max_depth=100,
                                 bootstrap=True,
                                 oob_score=False,
                                 n_jobs=-1,
                                 random_state=0,
                                 verbose=0,
                                 warm_start=False,
                                 class_weight='balanced')
    # Train the Classifier to take the training features and learn how they relate
    # to the training y (the species)
    clf.fit(X_Train.values, Y_Train.values)

    # Apply the Classifier we trained to the test data (which, remember, it has never seen before)
    preditected = clf.predict(X_Yest.values)


    # View the predicted probabilities of the first 10 observations
    probs = clf.predict_proba(X_Yest.values)
    print((probs[:, 1]))

    np.savetxt('prediction.txt', (probs[:, 1]), fmt="%s", newline="\n")


    # Feature Scaling
    from sklearn.preprocessing import StandardScaler

    # from sklearn.ensemble import RandomForestClassifier
    # classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
    # classifier.fit(X_Train, Y_Train)
    #
    # # Predicting the test set results
    # y_pred = classifier.predict(X_Test)
    #
    # # Making the Confusion Matrix
    #
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(Y_Test, y_pred)
    # print(cm)


import sys
if __name__ == '__main__':
    train_set = sys.argv[1]
    test_set = sys.argv[2]
    GridSearchCVforRandomForest(train_set, test_set)










dataFrame = loadDataSet(train_set)
# dataFrame = loadDataSet("trainingdataset.tsv")

# print(dataFrame)
# Split-out validation dataset
X = dataFrame.iloc[:, :71]
y = dataFrame.iloc[:, -1]
'''class distribution'''
print(dataFrame.groupby(y).size())


validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
scorings = ['average_precision', 'precision', 'recall', 'accuracy']


models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RANDOM FOREST', RandomForestClassifier(n_estimators=110,
                                 criterion='gini',
                                 min_samples_split=8,
                                 min_samples_leaf=3,
                                 min_weight_fraction_leaf=0.0,
                                 max_features='auto',
                                 max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,
                                 min_impurity_split=None,
                                 max_depth=100,
                                 bootstrap=True,
                                 oob_score=False,
                                 n_jobs=-1,
                                 random_state=0,
                                 verbose=0,
                                 warm_start=False,
                                 class_weight='balanced')))

# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NBayes', GaussianNB()))
models.append(('SVM', SVC(cache_size=200,
                          class_weight='balanced',
                          coef0=0.0,
                          decision_function_shape='ovr',
                          degree=3, gamma='auto',
                          kernel='rbf',
                          max_iter=-1,
                          probability=False,
                          random_state=10,
                          shrinking=True,
                          tol=0.001,
                          verbose=False)))


# evaluate each model in turn
results = []
outputs = {
    "KNN": "",
    "RF": "",
    "SVM": ""
}
names = []
# for name, model in models:
for model in models:
    model_name = "For %s: " % (model[0])
    print(model_name)
    print("-------------------------------")
    precisions = []
    recalls = []
    for score in scorings:
        kfold = StratifiedKFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model[1], X_train.values, Y_train.values, cv=kfold, scoring=score)
        if (score == "accuracy"):
            results.append(cv_results)
            #print("sample:", repr(cv_results))
            names.append(model[0])
            print("Accuracy: %.3f%% (%.3f%%)" % (cv_results.mean() * 100.0, cv_results.std() * 100.0))

        msg = "%s: (%f)" % (score, cv_results.mean())
        print(msg)


# show the plot

# # Compare Algorithms
# fig = pyplot.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(names)
# pyplot.show()













from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from matplotlib import pyplot
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, \
    precision_score, confusion_matrix, f1_score, average_precision_score


pyplot.style.use("ggplot")

'''
    LOAD THE DATASET 
    :param - data filename
============================================================================'''
# loading the dataset
def loadDataSet(datasetFileName):
    from numpy import genfromtxt
    sample = pd.read_csv(datasetFileName, sep='\t',header=None, dtype=None)
    # sample = sample[:20]
    #print(sample.columns)
    #print("dimension of diabetes data: {}".format(sample.shape))
    return sample



'''
    SPLITTING DATASET
============================================================================'''

def GridSearchCVforKNN():
    dataFrame = loadDataSet(train_set)
    dataFrameTest = loadDataSet(test_set)
    # print(dataFrame)
    X = dataFrame.iloc[:,:71]
    y = dataFrame.iloc[:,-1]


    # define the parameter values that should be searched
    # for python 2, k_range = range(1, 31)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 4, stratify=y)

    # print('y_train class distribution')
    # print(y_train.value_counts(normalize=True))
    # print('y_test class distribution')
    # print(y_test.value_counts(normalize=True))


    # list of scores from k_range
    k_scores = []


    # define the parameter values that should be searched
    # for python 2, k_range = range(1, 31)
    k_range = list(range(1, 40))


    weight_options = ["uniform", "distance"]
    param_grid = dict(n_neighbors=k_range, weights=weight_options)

    skf = StratifiedKFold(n_splits=10)
    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=skf, refit='precision_score', scoring='accuracy', return_train_score=True)
    grid.fit(X_train.values, y_train.values)

    # make the predictions
    y_pred = grid.predict(X_test.values)


    print('Best params for {}'.format('precision_score'))
    print(grid.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of KNN optimized for {} on the test data:'.format('precision_score'))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

    #print(round(grid.best_score_, 2))
    #print(grid.grid_scores_)


    y_scores = grid.predict_proba(X_test)[:, 1]
    # for classifiers with decision_function, this achieves similar results
    # y_scores = classifier.decision_function(X_test)
    p, r, thresholds = precision_recall_curve(y_test, y_scores)

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    average_precision = np.mean(precision)
    average_recall = np.mean(recall)

    print("Precision: " + repr(average_precision) + "Recall: " + repr(average_recall))

    # calculate F1 score
    f1 = f1_score(y_test, y_pred)
    # calculate precision-recall AUC
    from sklearn.metrics import auc
    auc = auc(recall, precision)
    # calculate average precision score
    ap = average_precision_score(y_test, y_scores)
    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))

    # plot no skill
    pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the roc curve for the model
    pyplot.plot(recall, precision, marker='.')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the plot
    pyplot.show()

# GridSearchCVforKNN()


#
# def GridSearchCVforRandomForest():
#     dataFrame = loadDataSet("trainingdataset_demo.tsv")
#     # print(dataFrame)
#     X = dataFrame.iloc[:, :71]
#     y = dataFrame.iloc[:, -1]
#
#     # Splitting the dataset into the Training set and Test set
#     from sklearn.model_selection import train_test_split
#     X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.25, random_state=999, stratify=y)
#
#     # Feature Scaling
#     from sklearn.preprocessing import StandardScaler
#
#     from sklearn.ensemble import RandomForestClassifier
#     classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
#     classifier.fit(X_Train, Y_Train)
#
#     # Predicting the test set results
#     y_pred = classifier.predict(X_Test)
#
#     # Making the Confusion Matrix
#
#     from sklearn.metrics import confusion_matrix
#     cm = confusion_matrix(Y_Test, y_pred)
#     print(cm)
#
#
# GridSearchCVforRandomForest()

from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')

dataFrame = loadDataSet(train_set)
dataFrame_test = loadDataSet(test_set)
# print(dataFrame)
# X_train = dataFrame.iloc[:,:71]
# y_train = dataFrame.iloc[:,-1]
# X_test = dataFrame_test.iloc[:,:71]

X = dataFrame.iloc[:,:71]
y = dataFrame.iloc[:,-1]


# define the parameter values that should be searched
# for python 2, k_range = range(1, 31)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 4, stratify=y)


clf = RandomForestClassifier(n_jobs=-1)

# param_grid = {
#     'min_samples_split': [3, 5, 10],
#     'n_estimators' : [100, 300],
#     'max_depth': [3, 5, 15, 25],
#     'max_features': [3, 5, 10, 20]
# }
param_grid = {
    'bootstrap': [True],
    'max_depth': [100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200]
}



def grid_search_wrapper(score):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf, param_grid,
                           cv=skf, return_train_score=True, scoring=score, n_jobs=-1)
    grid_search.fit(X_train.values, y_train.values)

    # make the predictions
    y_pred = grid_search.predict(X_test.values)
    probs = grid_search.predict_proba(X_test.values)
    #print(probs)

    print('Best params for {RandomForest}: ')
    print(grid_search.best_params_)
    #result = grid_search.cv_results_['mean_test_score']
    #print("Best Score: " + repr(result))

    print("----------------------------")

scorers = ['accuracy']

for x in range(len(scorers)):
    grid_search_wrapper(scorers[x])





# y_scores = grid_search_clf.predict_proba(X_test)[:, 1]
# for classifiers with decision_function, this achieves similar results
# y_scores = classifier.decision_function(X_test)





def GridSearchCVforSVM():
    dataFrame = loadDataSet(train_set)
    dataFrameTest = loadDataSet(test_set)
    # print(dataFrame)
    X = dataFrame.iloc[:,:71]
    y = dataFrame.iloc[:,-1]

    nfolds = StratifiedKFold(n_splits=10)

    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    print(grid_search.best_params_)
    # return grid_search.best_params_





GridSearchCVforSVM()


