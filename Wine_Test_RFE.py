from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd 

# load csv file
df = pd.read_csv("./Wine_Test_01/Wine_Test_01.csv")

# separate X and y
X = df.drop("quality", axis=1)
y = df["quality"]

# split into test/train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)

# a) estimator for RFE
estimator= SVC(kernel='linear', C=1000, gamma=0.1)

# b) RFE to select top 8 features
selector = RFE(estimator, n_features_to_select=8)
selector.fit(X_train, y_train)

# c) display the selected features 
selected_features = X_train.columns[selector.support_]
print("Selected Features: ", selected_features.tolist())

# d) reducing to selected features
X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]

# performing gridsearch with 10 hyperparams
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear'] 
}

grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train_reduced, y_train)

print("Best hyperparameters: ",grid.best_params_)

y_pred = grid.predict(X_test_reduced)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set: ", accuracy) 