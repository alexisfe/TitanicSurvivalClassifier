__author__ = 'alexis'

import numpy as np
import pandas as pd

def get_testing_errors(X_test, y_test, y_pred):

    y_test_np = np.array(y_test, dtype=pd.Series)

    y_comp = np.logical_and(y_pred, y_test_np)

    y_comp_idx = np.where(y_comp == False)

    X_test.iloc[y_comp_idx]
    y_test.iloc[y_comp_idx]

    test_errors = pd.concat([X_test, y_test], axis=1)

    test_errors.to_csv("error/test_errors.csv", index=False)