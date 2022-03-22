
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

NMSE = np.zeros(shape=(len(list_lamda), K))
RMSE = np.zeros(shape=(len(list_lamda),1))

for i in range(len(list_lamda)):
    cv_def = ms.RepeatedKFold(n_splits=K, n_repeats=1, random_state=None)
    # model uses RidgeCV for leave one out
    # Note: no feature transformation
    ridge_model = lm.RidgeCV(alphas=list_lamda[i])
    ridge_model.fit(X_train, y_train)
    # RMSE on the left-out folds
    NMSE[i, :] = ms.cross_val_score(ridge_model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv_def, n_jobs=1)
    RMSE[i] = np.sqrt(np.mean(abs(NMSE[i, :])))

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

print("RMSE corresponding to each lamda \n", RMSE)

# Write to CSV
file = 'C:/Users/Lannan Jiang/PycharmProjects/IML_Task1/submission.csv'
pd.DataFrame(RMSE).to_csv(file, index=False, header=None)
