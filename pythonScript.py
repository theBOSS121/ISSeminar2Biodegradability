# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# %% [markdown]
# ### Importing training and testing data sets, exploratory analysis of features

# %%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head() # display a few samples

# %% [markdown]
# ### Visualization of the feature space

# %%
f = plt.figure(figsize=(15, 15))

for i, col in enumerate(train.columns):
    print(col, len(train[col].unique()), end="; ")
    f.add_subplot(7, 6, i+1)
    plt.title(col)
    plt.ylabel(col)
    plt.xlabel("indexes")
    plt.tight_layout()
    plt.plot(train[col][train['Class'] == 1], "bo", markersize="1")
    plt.plot(train[col][train['Class'] == 2], "ro", markersize="1")

# %%
f = plt.figure(figsize=(10, 8))

b = [26, 35, 38, 0]
a = train.columns[b]

counter = 1
for i, col in enumerate(a):
    for j, col2 in enumerate(a[i+1:]):
        f.add_subplot(2, 3, counter)
        plt.title(col + " " + col2)
        plt.ylabel(col2)
        plt.xlabel(col)
        plt.tight_layout()
        plt.plot(train[col][train['Class'] == 1], train[col2][train['Class'] == 1], "bo", markersize="2")
        plt.plot(train[col][train['Class'] == 2], train[col2][train['Class'] == 2], "ro", markersize="2")
        counter += 1

plt.show()

# %%
f = plt.figure(figsize=(15, 15))

for i, col in enumerate(train.columns):
    f.add_subplot(7, 6, i+1)
    plt.title(col)
    plt.tight_layout()
    plt.boxplot(train[col])

# %% [markdown]
# 2.1 Exploration
# 
# Inspect the dataset. How balanced is the target variable? Are there any missing values present? If there
# are, choose a strategy that takes this into account.
# Most of your data is of the numeric type. Can you identify, by adopting exploratory analysis, whether
# some features are directly related to the target? What about feature pairs? Produce at least three types of
# visualizations of the feature space and be prepared to argue why these visualizations were useful for your
# subsequent analysis.

# %% [markdown]
# Target variable distributions in test and training sets are close to [2/3 1/3].
# 
# Yes, there are some missing values. One possible strategy is to drop those that don't have all attribute values. Some classifiers however don't really need all information, so you can just ignore missing rows for that specific attributes.
# 
# The visualizations are above. We haven't found anything very concrete. Some of the features are more and some less releted to target. Random forest feature importances helped us to know which attributes are less/more important

# %% [markdown]
# ### Majority classifier

# %%
majority = train['Class'].value_counts()
majorityArr = np.array(majority)
print("Class distribution (train)")
print(majority)
print("Percentage:")
print(np.array(majorityArr[0] / np.sum(majorityArr)))

majorityTest = test['Class'].value_counts()
majorityTestArr = np.array(majorityTest)
print("Class distribution (test)")
print(majorityTest)
print("Percentage:")
print(np.array(majorityTestArr[0] / np.sum(majorityTestArr)))

# %% [markdown]
# ### Random classifier
# 
# There are two classes if we choose between them randomly accuracy is 1/2.

# %% [markdown]
# ### Preprocessing
# Removing data that doesn't have all values, seperating features and target variable, choosing subsets of features. Some other possible preprocessing would be normalization of attributes, setting NA values to average/max/min/random/... values, making new features using linear/non-linear combinations of two or more attributes. Some preprocessing techniques did not improve our results so we deleted some of the code for them.

# %%
# removing NA values from train dataframe
train = train.dropna()
# Separate input features (X) and target variable (y)
y = train.Class
X = train.drop('Class', axis=1)
testY = test.Class
testX = test.drop('Class', axis=1)

# drop bad/uninformative columns
# dropColumns = np.array(['V26', 'V29', 'V19', 'V21', 'V24', 'V20', 'V4'])
# X = X.drop(dropColumns, axis=1)
# testX = testX.drop(dropColumns, axis=1)

# %% [markdown]
# ### Decision tree

# %%
clf_decitionTree = DecisionTreeClassifier(random_state=0)
clf_decitionTree.fit(X, y)

pred_y_decitionTree = clf_decitionTree.predict(testX)
print(accuracy_score(testY, pred_y_decitionTree))
prob_y_decisionTree = clf_decitionTree.predict_proba(testX)
prob_y_decisionTree = [p[1] for p in prob_y_decisionTree]
print(roc_auc_score(testY, prob_y_decisionTree))
print(f1_score(testY, pred_y_decitionTree))
print(precision_score(testY, pred_y_decitionTree))
print(recall_score(testY, pred_y_decitionTree))

# %% [markdown]
# ### KNN

# %%
clf_knn = KNeighborsClassifier(n_neighbors=9)
clf_knn.fit(X, y)

pred_y_knn = clf_knn.predict(testX)
print(accuracy_score(testY, pred_y_knn))
prob_y_knn = clf_knn.predict_proba(testX)
prob_y_knn = [p[1] for p in prob_y_knn]
print(roc_auc_score(testY, prob_y_knn))
print(f1_score(testY, pred_y_knn))
print(precision_score(testY, pred_y_knn))
print(recall_score(testY, pred_y_knn))

# %% [markdown]
# ### SVC / SVM

# %%
clf_svc = SVC(kernel='linear',  class_weight='balanced', probability=True)
clf_svc.fit(X, y)

pred_y_svc = clf_svc.predict(testX)
print(accuracy_score(testY, pred_y_svc))
prob_y_svc = clf_svc.predict_proba(testX)
prob_y_svc = [p[1] for p in prob_y_svc]
print(roc_auc_score(testY, prob_y_svc))
print(f1_score(testY, pred_y_svc))
print(precision_score(testY, pred_y_svc))
print(recall_score(testY, pred_y_svc))

# %% [markdown]
# ### Random forest

# %%
clf_randomForest = RandomForestClassifier(random_state=1234)
clf_randomForest.fit(X, y)
pred_y_randomForest = clf_randomForest.predict(testX)
print(accuracy_score(testY, pred_y_randomForest))
prob_y_randomForest = clf_randomForest.predict_proba(testX)
prob_y_randomForest = [p[1] for p in prob_y_randomForest]
print(roc_auc_score(testY, prob_y_randomForest))
print(f1_score(testY, pred_y_randomForest))
print(precision_score(testY, pred_y_randomForest))
print(recall_score(testY, pred_y_randomForest))

f = plt.figure(figsize=(8, 8))
sorted_idx = clf_randomForest.feature_importances_.argsort()
plt.barh(train.columns[sorted_idx], clf_randomForest.feature_importances_[sorted_idx])

# %% [markdown]
# ### Ada boost

# %%
clf_adaboost = AdaBoostClassifier(n_estimators = 50, learning_rate = 0.2)
clf_adaboost.fit(X, y)

pred_y_adaboost = clf_adaboost.predict(testX)
print(accuracy_score(testY, pred_y_adaboost))
prob_y_adaboost = clf_adaboost.predict_proba(testX)
prob_y_adaboost = [p[1] for p in prob_y_adaboost]
print(roc_auc_score(testY, prob_y_adaboost))
print(f1_score(testY, pred_y_adaboost))
print(precision_score(testY, pred_y_adaboost))
print(recall_score(testY, pred_y_adaboost))

# %%
def printBestScoreOnTest(scores):
    scoresOnTestForEachEstimator = np.array([])
    for e in scores['estimator']:
        pred_y = e.predict(testX)
        ac = accuracy_score(testY, pred_y)
        prob_y = e.predict_proba(testX)
        prob_y = [p[1] for p in prob_y]
        roc = roc_auc_score(testY, prob_y)
        f1 = f1_score(testY, pred_y)
        precision = precision_score(testY, pred_y)
        recall = recall_score(testY, pred_y)
        temp = np.array([ac, roc, f1, precision, recall])
        if scoresOnTestForEachEstimator.size == 0:
            scoresOnTestForEachEstimator = temp
        else:
            scoresOnTestForEachEstimator = np.vstack((scoresOnTestForEachEstimator, temp))
    bestIndex = np.argmax(scoresOnTestForEachEstimator[:, 0])
    return scoresOnTestForEachEstimator[bestIndex, :]

# %%
def printScores(scores):
    avgAccuracy = np.mean(scores['test_accuracy'])
    stdAccuracy = np.std(scores['test_accuracy'])
    avgF1 = np.mean(scores['test_f1_macro'])
    stdF1 = np.std(scores['test_f1_macro'])
    avgRecall = np.mean(scores['test_recall_macro'])
    stdRecall = np.std(scores['test_recall_macro'])
    avgAUC = np.mean(scores['test_roc_auc'])
    stdAUC = np.std(scores['test_roc_auc'])
    print(f"Precision average {avgAccuracy} with standard deviation {stdAccuracy}")
    print(f"F1 average {avgF1} with standard deviation {stdF1}")
    print(f"Recall average {avgRecall} with standard deviation {stdRecall}")
    print(f"AUC average {avgAUC} with standard deviation {stdAUC}")
    scoresTestArr = printBestScoreOnTest(scores)
    print(f'Test [accuracy, roc AUC, f1, precision, recall]')
    print(scoresTestArr)
    return avgAccuracy, avgF1, avgRecall, avgAUC, scoresTestArr

# %%
scoring = ['accuracy', 'f1_macro', 'recall_macro', 'roc_auc']
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1234)

print("DecisionTree:")
scores = cross_validate(clf_decitionTree, X, y, cv=rkf.split(X), scoring=scoring, return_estimator=True)
accDT, f1DT, recallDT, aucDT, testArrDT = printScores(scores)
print("KNN:")
scores = cross_validate(clf_knn, X, y, cv=rkf.split(X), scoring=scoring, return_estimator=True)
accKNN, f1KNN, recallKNN, aucKNN, testArrKNN = printScores(scores)
print("SVC:")
scores = cross_validate(clf_svc, X, y, cv=rkf.split(X), scoring=scoring, return_estimator=True)
accSVC, f1SVC, recallSVC, aucSVC, testArrSVC = printScores(scores)
print("Random forest:")
scores = cross_validate(clf_randomForest, X, y, cv=rkf.split(X), scoring=scoring, return_estimator=True)
accRF, f1RF, recallRF, aucRF, testArrRF = printScores(scores)
print("Ada boost:")
scores = cross_validate(clf_adaboost, X, y, cv=rkf.split(X), scoring=scoring, return_estimator=True)
accAB, f1AB, recallAB, aucAB, testArrAB = printScores(scores)

accArr = np.array([accDT, accKNN, accSVC, accRF, accAB])
f1Arr = np.array([f1DT, f1KNN, f1SVC, f1RF, f1AB])
recallArr = np.array([recallDT, recallKNN, recallSVC, recallRF, recallAB])
aucArr = np.array([aucDT, aucKNN, aucSVC, aucRF, aucAB])

testScoresArr = np.array([testArrDT, testArrKNN, testArrSVC, testArrRF, testArrAB])

# %%
print("Training")
xAxis = ['DecisionTree', 'KNN', 'SVC', 'RandomForest', 'AdaBoost']
f = plt.figure(figsize=(10, 10))
f.add_subplot(2, 2, 1)
plt.title("Precision")
plt.ylabel("Accuracy")
plt.xlabel("Classifiers")
plt.tight_layout()
plt.plot(xAxis, accArr, "bo", markersize="5")

f.add_subplot(2, 2, 2)
plt.title("F1")
plt.ylabel("f1")
plt.xlabel("Classifiers")
plt.tight_layout()
plt.plot(xAxis, f1Arr, "bo", markersize="5")

f.add_subplot(2, 2, 3)
plt.title("Recall")
plt.ylabel("Recall")
plt.xlabel("Classifiers")
plt.tight_layout()
plt.plot(xAxis, recallArr, "bo", markersize="5")

f.add_subplot(2, 2, 4)
plt.title("AUC")
plt.ylabel("AUC")
plt.xlabel("Classifiers")
plt.tight_layout()
plt.plot(xAxis, aucArr, "bo", markersize="5")

# %%
print("Test")
xAxis = ['DecisionTree', 'KNN', 'SVC', 'RandomForest', 'AdaBoost']
f = plt.figure(figsize=(10, 10))
f.add_subplot(3, 2, 1)
plt.title("Precision")
plt.ylabel("Accuracy")
plt.xlabel("Classifiers")
plt.tight_layout()
plt.plot(xAxis, testScoresArr[:, 3], "bo", markersize="5")

f.add_subplot(3, 2, 2)
plt.title("F1")
plt.ylabel("f1")
plt.xlabel("Classifiers")
plt.tight_layout()
plt.plot(xAxis, testScoresArr[:, 2], "bo", markersize="5")

f.add_subplot(3, 2, 3)
plt.title("Recall")
plt.ylabel("Recall")
plt.xlabel("Classifiers")
plt.tight_layout()
plt.plot(xAxis, testScoresArr[:, 4], "bo", markersize="5")

f.add_subplot(3, 2, 4)
plt.title("AUC")
plt.ylabel("AUC")
plt.xlabel("Classifiers")
plt.tight_layout()
plt.plot(xAxis, testScoresArr[:, 1], "bo", markersize="5")

f.add_subplot(3, 2, 5)
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Classifiers")
plt.tight_layout()
plt.plot(xAxis, testScoresArr[:, 0], "bo", markersize="5")


# %% [markdown]
# Comment on the performance of algorithms and visualize their final scores. How do they perform against
# the random baseline? What about the constant one? How do different learning scenarios impact the final
# score? Are the differences between the models statistically significant?
# 
# Random forest, SVC and AdaBoost performed better than others. DecisionTree and KNN performed significantly worse than others. They all perform better than random and constant baseline (50%/66% accuracy). Different parameters will change final scores and also different preprocessing of data might improve the final score (less noise/outliers, more accurate data,...). Differences of the RandomForest, SVC, AdaBoost are not very big, but depending on the needs of the classification different classification model might be more desirable.

# %% [markdown]
# ### Paramaters tuning (Random forest)

# %%
params = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)], # Number of trees in random forest
               'max_features': ['sqrt'], # Number of features to consider at every split
               'max_depth': [2,4], # Maximum number of levels in tree
               'min_samples_split':  [2, 5], # Minimum number of samples required to split a node
               'min_samples_leaf': [1, 2], # Minimum number of samples required at each leaf node
               'bootstrap': [True, False]} # Method of selecting samples for training each tree
rf_Model = RandomForestClassifier()
rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = params, cv = 5, verbose=2, n_jobs = 4, scoring='f1')
rf_Grid.fit(X, y)
rf_Grid.best_params_
rf_Random = RandomizedSearchCV(estimator = rf_Model, param_distributions = params, cv = 5, verbose=2, n_jobs = 4, scoring='f1')
rf_Random.fit(X, y)
rf_Random.best_params_

# %%
print (f'Train Accuracy - : {rf_Grid.score(X,y):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(testX,testY):.3f}')
print (f'Train Accuracy - : {rf_Random.score(X,y):.3f}')
print (f'Test Accuracy - : {rf_Random.score(testX,testY):.3f}')

print(f'Test accuracy score, roc AUC, f1, percision, recall with the best estimator (model):')
pred_y_randomForest = rf_Random.best_estimator_.predict(testX)
print(accuracy_score(testY, pred_y_randomForest))
prob_y_randomForest = clf_randomForest.predict_proba(testX)
prob_y_randomForest = [p[1] for p in prob_y_randomForest]
print(roc_auc_score(testY, prob_y_randomForest))
print(f1_score(testY, pred_y_randomForest))
print(precision_score(testY, pred_y_randomForest))
print(recall_score(testY, pred_y_randomForest))

# %% [markdown]
# ### Parameter tuning (AdaBoost)

# %%
adaBoostModel = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

params = {'base_estimator__max_depth':[i for i in range(2,11,2)],
              'base_estimator__min_samples_leaf':[5,10],
              'n_estimators':[10,50,250,1000],
              'learning_rate':[0.01,0.1]}

adaBoost_Grid = GridSearchCV(adaBoostModel, params, verbose=3, cv=5, scoring='f1', n_jobs=-1)
adaBoost_Grid.fit(X,y)
adaBoost_Grid.best_params_
adaBoost_Random = RandomizedSearchCV(estimator = adaBoostModel, param_distributions = params, cv = 5, verbose=2, n_jobs = 4, scoring='f1')
adaBoost_Random.fit(X, y)
adaBoost_Random.best_params_

# %%
print (f'Train Accuracy - : {adaBoost_Grid.score(X,y):.3f}')
print (f'Test Accuracy - : {adaBoost_Grid.score(testX,testY):.3f}')
print (f'Train Accuracy - : {adaBoost_Random.score(X,y):.3f}')
print (f'Test Accuracy - : {adaBoost_Random.score(testX,testY):.3f}')

print(f'Test accuracy score, roc AUC, f1, percision, recall with the best estimator (model):')
pred_y_adaboost = adaBoost_Grid.best_estimator_.predict(testX)
print(accuracy_score(testY, pred_y_adaboost))
prob_y_adaboost = clf_adaboost.predict_proba(testX)
prob_y_adaboost = [p[1] for p in prob_y_adaboost]
print(roc_auc_score(testY, prob_y_adaboost))
print(f1_score(testY, pred_y_adaboost))
print(precision_score(testY, pred_y_adaboost))
print(recall_score(testY, pred_y_adaboost))


# %% [markdown]
# ### Parameter tuning (SVC)

# %%
from scipy import stats

svcModel = SVC(kernel='linear',  class_weight='balanced', probability=True)
params = {"C": np.arange(2, 10, 2),
             "gamma": np.arange(0.1, 1, 0.2)}
 
svc_Grid = GridSearchCV(svcModel, param_grid = params, n_jobs = 4, cv = 5, scoring='f1') 
svc_Grid.fit(X, y) 
svc_Grid.best_params_
 
params = {"C": stats.uniform(2, 10),
            "gamma": stats.uniform(0.1, 1)}
              
svc_Random = RandomizedSearchCV(svcModel, param_distributions = params, n_iter = 20, n_jobs = 4, cv = 5, random_state = 2017, scoring='f1') 
svc_Random.fit(X, y) 
svc_Random.best_params_

# %%
print (f'Train Accuracy - : {svc_Grid.score(X,y):.3f}')
print (f'Test Accuracy - : {svc_Grid.score(testX,testY):.3f}')
print (f'Train Accuracy - : {svc_Random.score(X,y):.3f}')
print (f'Test Accuracy - : {svc_Random.score(testX,testY):.3f}')


print(f'Test accuracy score, roc AUC, f1, percision, recall with the best estimator (model):')
pred_y_svc = svc_Random.best_estimator_.predict(testX)
print(accuracy_score(testY, pred_y_svc))
prob_y_svc = clf_svc.predict_proba(testX)
prob_y_svc = [p[1] for p in prob_y_svc]
print(roc_auc_score(testY, prob_y_svc))
print(f1_score(testY, pred_y_svc))
print(precision_score(testY, pred_y_svc))
print(recall_score(testY, pred_y_svc))


