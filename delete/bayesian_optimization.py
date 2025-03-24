import numpy as np
import pdb
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

# Define the discrete search space (example: integers from 0 to 10 for each variable)
search_space = {
    "x1": np.arange(0, 11),
    "x2": np.arange(0, 11)
}
search_space_list = np.array(np.meshgrid(search_space["x1"], search_space["x2"])).T.reshape(-1, 2)

# Define the objective function to minimize
def objective_function(x):
    x1, x2 = x
    return (x1 - 5) ** 2 + (x2 - 7) ** 2  # Example: quadratic function

# Initialize random samples
np.random.seed(42)
num_initial_samples = 10
initial_indices = np.random.choice(len(search_space_list), num_initial_samples, replace=False)
X_train = search_space_list[initial_indices]
y_train = np.array([objective_function(x) for x in X_train])

# Train the surrogate model (Random Forest)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Acquisition function (greedy selection based on predicted mean)
def acquisition_function(X_candidates, model):
    preds = model.predict(X_candidates)
    return preds  # Lower is better (minimization)

# Bayesian Optimization Loop
num_iterations = 10
for i in range(num_iterations):
    # Select the next point using the acquisition function (greedy selection)
    X_candidates = np.array([x for x in search_space_list if x.tolist() not in X_train.tolist()])
    pdb.set_trace()
    if len(X_candidates) == 0:
        break
    y_pred = acquisition_function(X_candidates, rf)
    next_point = X_candidates[np.argmin(y_pred)]

    # Evaluate the function at the new point
    y_new = objective_function(next_point)

    # Update the dataset
    X_train = np.vstack((X_train, next_point))
    y_train = np.append(y_train, y_new)

    # Retrain the surrogate model
    rf.fit(X_train, y_train)

    print(f"Iteration {i+1}: Selected {next_point}, Function Value: {y_new}")

# Return the best found solution
best_index = np.argmin(y_train)
best_x = X_train[best_index]
best_y = y_train[best_index]
print(f"\nBest solution found: {best_x}, Function Value: {best_y}")
