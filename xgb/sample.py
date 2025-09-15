import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# load the dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix (optimized data structure for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters
params = {
    "objective": "binary:logistic",  # for classification
    "eval_metric": "logloss",        # evaluation metric
    "eta": 0.1,                      # learning rate
    "max_depth": 4,                  # max tree depth
    "lambda": 1,                     # L2 regularization
    "alpha": 0                       # L1 regularization
}

# Train model
bst = xgb.train(params, dtrain, num_boost_round=100)

# Predict
y_pred = bst.predict(dtest)
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]

# Evaluate
acc = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", acc)


# Explanation for L1 and L2 regularization:
# L1 regularization (alpha) adds a penalty equal to the absolute value of the magnitude of coefficients.
# It can lead to sparse models where some feature weights are zero, effectively performing feature selection.   
# L2 regularization (lambda) adds a penalty equal to the square of the magnitude of coefficients.
# It discourages large coefficients but does not enforce sparsity, leading to smaller, more evenly distributed weights.