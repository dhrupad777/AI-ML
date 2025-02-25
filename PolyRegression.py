import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Replace this with your own data loading mechanism
def load_house_data():
    # Example data: size(sqft), bedrooms, floors, age
    X_train = np.array([[2104, 5, 1, 45],
                        [1416, 3, 2, 40],
                        [852, 2, 1, 35],
                        [1534, 3, 2, 30]])
    y_train = np.array([460, 232, 178, 220])
    return X_train, y_train

X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train, axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_train_norm, axis=0)}")

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [1000, 2000, 3000],
    'eta0': [0.01, 0.1, 0.2],
    'penalty': ['l2', 'l1', 'elasticnet']
}

sgdr = SGDRegressor()
grid_search = GridSearchCV(estimator=sgdr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_norm, y_train)

best_sgdr = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Train the best model
best_sgdr.fit(X_train_norm, y_train)
print(f"number of iterations completed: {best_sgdr.n_iter_}, number of weight updates: {best_sgdr.t_}")

b_norm = best_sgdr.intercept_
w_norm = best_sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")

# Make predictions
y_pred_train = best_sgdr.predict(X_train_norm)
y_pred_test = best_sgdr.predict(X_test_norm)

# Calculate error metrics
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Training set - MSE: {mse_train}, MAE: {mae_train}, R-squared: {r2_train}")
print(f"Testing set - MSE: {mse_test}, MAE: {mae_test}, R-squared: {r2_test}")

# Plot predictions and targets vs original features
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label='target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:, i], y_pred_train, color='orange', label='predict')
ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("Target versus Prediction using z-score normalized model")
plt.show()

# Plot learning curve
def plot_learning_curve(estimator, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("MSE")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

plot_learning_curve(best_sgdr, X_train_norm, y_train, cv=3, n_jobs=-1)
plt.show()