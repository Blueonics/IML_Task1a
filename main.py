
# Task 1a
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm
from sklearn import model_selection as ms


# Path to train data
df = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task1/train.csv')

# training data inputs
X = df.loc[:, 'x1':'x13']
y = df.loc[:, 'y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

K = 10
list_lamda = [0.1, 1, 10, 100, 200]

cv_def = ms.RepeatedKFold(n_splits=K, random_state=None)
# model uses RidgeCV for leave one out
ridge_model = lm.RidgeCV(alphas=list_lamda)

# RMSE on the left-out folds
NMSE = ms.cross_val_score(ridge_model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv_def, n_jobs=1)
RMSE = np.sqrt(np.mean(abs(NMSE)))

# Note: no feature transformation
# linear regression
lin_reg = lm.LinearRegression()
lin_reg.fit(X_train, y_train)

# predict on test set
y_pred_arr = lin_reg.predict(X_test)

# print("My test y ", y_test)
# print("My predicted y ", y_pred)

y_test_arr = np.asarray(y_test)

loss = np.sqrt(np.mean((y_test_arr - y_pred_arr)**2))
print("We have an RMSE between y predictions and y test data ", loss)

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':




