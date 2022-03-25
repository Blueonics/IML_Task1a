
# Task 1a
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn import model_selection as ms


# Path to train data
df = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task1/train.csv')

# training data inputs
X = np.asarray(df.loc[:, 'x1':'x13'])
y = np.asarray(df.loc[:, 'y'])

K = 10
list_lamda = [0.1, 1, 10, 100, 200]

NMSE = np.zeros(shape=(len(list_lamda), K))
RMSE = np.zeros(shape=(len(list_lamda), 1))

for i in range(len(list_lamda)):
    cv_def = ms.RepeatedKFold(n_splits=K, n_repeats=1, random_state=1)
    # model uses RidgeCV for leave one out
    # Note: no feature transformation
    ridge_model = lm.RidgeCV(alphas=list_lamda[i])
    # RMSE on the left-out folds
    NMSE[i, :] = ms.cross_val_score(ridge_model, X, y, scoring='neg_root_mean_squared_error', cv=cv_def, n_jobs=1)
    RMSE[i] = np.mean(abs(NMSE[i, :]))

print("RMSE corresponding to each lamda \n", RMSE)

# Write to CSV
file = 'C:/Users/Lannan Jiang/PycharmProjects/IML_Task1/submission.csv'
pd.DataFrame(RMSE).to_csv(file, index=False, header=False)
