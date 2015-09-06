__author__ = 'alexis'

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import validation_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.learning_curve import learning_curve

import extract_features as ef
import plot_learning_curve as plc
import get_testing_errors as gte


print "Reading dataset..."
ds = pd.read_csv('data/train.csv')
kaggle_ds = pd.read_csv('data/test.csv')

age_median = ds['Age'].median()

print "Engineering features..."
ef.transform(ds, age_median)
ef.transform(kaggle_ds, age_median)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength", "Title", "CabinType", "NumCabins", "FamilyName"]
predictors = ["Pclass", "Sex", "NameLength", "Title", "Fare"]

X_train, X_test, y_train, y_test = train_test_split(ds[predictors], ds["Survived"], test_size=0.2, random_state=0)

cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)

# #Feature selection
# selector = SelectKBest(f_classif, k=5)
# selector.fit(ds[predictors], ds["Survived"])
#
# # Get the raw p-values for each feature, and transform from p-values into scores
# scores = -np.log10(selector.pvalues_)
#
# # Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
# plt.bar(range(len(predictors)), scores)
# plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()


param = {'n_estimators': list(np.arange(10, 150, 10)), 'min_samples_split': list(np.arange(1, 10, 2)), 'min_samples_leaf': list(np.arange(1, 10, 2))}
rfc = RandomForestClassifier(n_estimators = 120, min_samples_split=5, min_samples_leaf=5)
# print "GridSearchCV on RFC..."
# rfc = GridSearchCV(estimator=rfc, cv=cv, param_grid=param)
rfc.fit(X_train, y_train)
# # summarize the results of the grid search
# print(rfc.best_score_)
# print "Best n_estimators found by GridSearch: ", rfc.best_estimator_.n_estimators
# print "Best min_samples_split found by GridSearch: ", rfc.best_estimator_.min_samples_split
# print "Best min_samples_leaf found by GridSearch: ", rfc.best_estimator_.min_samples_leaf

title = "Learning curves (Random Forest Classifier)"

plc.plot_learning_curve(rfc, title, X_train, y_train, cv=cv)
plt.show()

print "Prediction score on test set: ", rfc.score(X_test, y_test)

print "Creating testing errors file..."
y_pred = rfc.predict(X_test)
gte.get_testing_errors(X_test, y_test, y_pred)

print "Creating kaggle submission file..."
predictions = rfc.predict(kaggle_ds[predictors])
submission = pd.DataFrame({"PassengerId": kaggle_ds["PassengerId"], "Survived": predictions})
submission.to_csv("submission/kaggle.csv", index=False)


