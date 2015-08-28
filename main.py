__author__ = 'alexis'

import pandas as pd

train = pd.read_csv('/Users/alexis/Downloads/train.csv')
test = pd.read_csv('/Users/alexis/Downloads/test.csv')

train['Age'] = train['Age'].fillna(train['Age'].median())
train['Age'] = train['Age'].fillna(train['Age'].median())


i = 0
for sex in train['Sex'].unique():
    train['Sex'].loc[train['Sex'] == sex, 'Sex'] = i
    i = i + 1


train['Embarked'] = train['Embarked'].fillna('S')

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

# Import the linear regression class
from sklearn.linear_model import LogisticRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LogisticRegression()
# Generate cross validation folds for the train dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(train.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (train[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = train["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(train[predictors].iloc[test,:])
    predictions.append(test_predictions)



