__author__ = 'alexis'

import re
import math

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, f_classif

default_titles = ["Mr", "Mrs", "Ms", "Miss", "Mx", "Master", "Maid", "Madam", "Madame", "Mlle", "Mme"]
academic_titles = ["Dr"]
aristocratic_titles = ["Hon", "Don", "Lady", "Countess", "Jonkheer", "Sir"]
military_titles = ["Major", "Col", "Capt"]
religious_titles = ["Rev"]

def get_ticket_class(ticket):
    ticket_search = re.search('[Aa](\.)*', ticket)

def get_family_name(name):
    family_search = re.search('([A-Za-z]+)\,', name)
    if family_search:
        return family_search.group(1)
    return ""

def get_cabin_type(cabin):
    if type(cabin) is float:
        if math.isnan(cabin):
            return "Z"

    return str(cabin)[0]

def get_num_cabins(cabin):
    if type(cabin) is float:
        if math.isnan(cabin):
            return 1

    return len(re.findall('[A-Z]', str(cabin)))

def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def get_num_tickets(ds):
    return
    #return ds.loc[ds["Ticket"] == ticket, "Ticket"].count()

def get_fare_per_person(ticket):
    return
    #ds["Ticket"]

def encode(ds):
    le = LabelEncoder()
    le.fit(ds)
    return le.transform(ds)

def transform(ds, age_median):
    ds['Age'] = ds['Age'].fillna(age_median)

    ds["Sex"] = encode(ds["Sex"])

    ds['Fare'] = ds['Fare'].fillna(ds['Fare'].median())
    ds['Fare'] = ds['Fare'].apply(get_fare)

    ds['Embarked'] = ds['Embarked'].fillna('S')
    ds["Embarked"] = encode(ds["Embarked"])

    ds["FamilySize"] = ds["SibSp"] + ds["Parch"]

    #Calculate name features
    ds["NameLength"] = ds["Name"].apply(lambda x: len(x))

    titles = ds["Name"].apply(get_title)
    ds["Title"] = encode(titles)

    family_names = ds["Name"].apply(get_family_name)
    ds["FamilyName"] = encode(family_names)

    #Calculate cabin features
    cabin_types = ds["Cabin"].apply(get_cabin_type)
    ds["CabinType"] = encode(cabin_types)

    ds["NumCabins"] = ds["Cabin"].apply(get_num_cabins)

    #Calculate ticket features
    #ds["NumTickets"] = get_num_tickets(ds)
    #ds["NumTickets"] = ds["Ticket"].apply()

    return ds

print "Reading training and testing sets..."
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

age_median = train['Age'].median()

print "Engineering features..."
transform(train, age_median)
transform(test, age_median)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength", "Title", "CabinType", "NumCabins", "FamilyName"]

#Feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(train[predictors], train["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

#predictors = ["Pclass", "Sex", "CabinType", "NameLength", "Title"]

print "Starting up algorithms..."

param = {'n_estimators': list(np.arange(10, 150, 10)), 'min_samples_split': list(np.arange(1, 10, 2)), 'min_samples_leaf': list(np.arange(1, 10, 2))}
rfc = RandomForestClassifier()
print "GridSearchCV on RFC..."
grid = GridSearchCV(estimator=rfc, param_grid=param)
grid.fit(train[predictors], train["Survived"])
# summarize the results of the grid search
print(grid.best_score_)
print "Best n_estimators found by GridSearch: ", grid.best_estimator_.n_estimators
print "Best min_samples_split found by GridSearch: ", grid.best_estimator_.min_samples_split
print "Best min_samples_leaf found by GridSearch: ", grid.best_estimator_.min_samples_leaf
#
gbc = GradientBoostingClassifier(random_state=1, n_estimators=20, max_depth=5)
#
# param = {'C': list(np.arange(0.01, 1, 0.01))}
# lr = LogisticRegression(random_state=1)
#
# print "GridSearchCV on LR..."
# grid = GridSearchCV(estimator=lr, param_grid=param)
# grid.fit(train[predictors], train["Survived"])
# # summarize the results of the grid search
# #print(grid.best_score_)
# print "Best C estimator found by GridSearch: ", grid.best_estimator_.C

# svc_params = {'C': np.arange(0.01, 1, 0.1)
#     , 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
#     }
# svc = SVC()
#
# print "GridSearchCV on SVC..."
# grid = GridSearchCV(estimator=svc, param_grid=svc_params)
# grid.fit(train[predictors], train["Survived"])
# print(grid)
# # summarize the results of the grid search
# print(grid.best_score_)
# print(grid.best_estimator_.alpha)
#
# alg = rfc
#
# # Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
# scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)
#
# # Take the mean of the scores (because we have one for each fold)
# print(scores.mean())
#
# alg = gbc
#
# # Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
# scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)
#
# # Take the mean of the scores (because we have one for each fold)
# print(scores.mean())

alg = rfc

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

alg.fit(train[predictors], train["Survived"])

predictions = alg.predict(test[predictors])

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})

submission.to_csv("submission/kaggle.csv", index=False)


