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

def transform(ds, age_median):
    ds['Age'] = ds['Age'].fillna(age_median)

    ds.loc[ds['Sex'] == 'male', 'Sex'] = 1
    ds.loc[ds['Sex'] == 'female', 'Sex'] = 2

    ds['Embarked'] = ds['Embarked'].fillna('S')
    ds['Fare'] = ds['Fare'].fillna(ds['Fare'].median())

    ds.loc[ds["Embarked"] == "S", "Embarked"] = 0
    ds.loc[ds["Embarked"] == "C", "Embarked"] = 1
    ds.loc[ds["Embarked"] == "Q", "Embarked"] = 2

    ds["FamilySize"] = ds["SibSp"] + ds["Parch"]
    ds["NameLength"] = ds["Name"].apply(lambda x: len(x))

    titles = ds["Name"].apply(get_title)

    titles_le = LabelEncoder()
    titles_le.fit(titles)
    ds["Title"] = titles_le.transform(titles)

    cabin_types = ds["Cabin"].apply(get_cabin_type)
    cabin_types_le = LabelEncoder()
    cabin_types_le.fit(cabin_types)
    ds["CabinType"] = cabin_types_le.transform(cabin_types)

    ds["NumCabins"] = ds["Cabin"].apply(get_num_cabins)

    family_names = ds["Name"].apply(get_family_name)
    family_names_le = LabelEncoder()
    family_names_le.fit(family_names)
    ds["FamilyName"] = family_names_le.transform(family_names)

    return ds

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

age_median = train['Age'].median()

transform(train, age_median)
transform(test, age_median)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength", "Title", "CabinType", "NumCabins", "FamilyName"]

# #Feature selection
# selector = SelectKBest(f_classif, k=5)
# selector.fit(train[predictors], train["Survived"])
#
# # Get the raw p-values for each feature, and transform from p-values into scores
# scores = -np.log10(selector.pvalues_)
#
# # Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
# plt.bar(range(len(predictors)), scores)
# plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()

predictors = ["Pclass", "Sex", "CabinType", "NameLength", "Title"]

rfc = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
gbc = GradientBoostingClassifier(random_state=1, n_estimators=20, max_depth=5)
lr = LogisticRegression(random_state=1)

svc_params = {'C': np.arange(0.01, 10, 0.05)
    , 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }
svc = SVC()

grid = GridSearchCV(estimator=svc, param_grid=svc_params)
grid.fit(train[predictors], train["Survived"])
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.alpha)

alg = rfc

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

alg = gbc

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

alg = lr

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

alg = svc

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


# alg.fit(train[predictors], train["Survived"])
#
# predictions = alg.predict(test[predictors])
#
# submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
#
# submission.to_csv("submission/kaggle.csv", index=False)


