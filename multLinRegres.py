# Multiple linear regression model for housing data
# 7/22/24
# Devan Parekh

import copy, math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

"""
Features(5): Sqft, # bedrooms, # bathrooms, lot size (acres), age

data.txt format: sqft bedrooms bathrooms lotsize age price
"""

def plot_features_vs_data(x, y):
    """
    Plots each feature individually against a shared price y-axis
    to help visualize feature spread and find outliers in the data

    args:
        x (numpy array with 5 cols) : used to plot features and their data
        y (numpy array with 1 col)  : used to plot price data
    
    returns: 
        none
    """
    fix, axs = plt.subplots(1, 5, figsize=(20,5), sharey=True)
    fix.suptitle("All features vs price")

    features = ["Sqft", "# bedrooms", "# bathrooms", "lot size", "age"]
    for i in range(5):
        axs[i].scatter(x[:,i], y, c='r', marker="x")
        axs[i].set_xlabel(features[i])
        axs[i].set_ylabel("Price")
        axs[i].set_title(f"{features[i]} vs price")

    plt.tight_layout()
    plt.show()

def normalize_x_data(x):
    """
    Normalizes the data using min-max

    args: 
        x (np array) : original x train data

    returns
        normalized_x : normalized x train data
    """
    normalized_x = np.zeros_like(x)
    for i in range(x.shape[1]):
        min = np.min(x[:, i])
        max = np.max(x[:,i])
        normalized_x[:,i] = (x[:,i] - min) / (max - min)
    return normalized_x


def cost_function(x, y, w, b):
    m = len(x)
    accumulated_cost = 0
    for i in range(m):
        f_wb_i = (np.dot(x[i], w) + b)
        err = f_wb_i - y[i]
        err = err ** 2
        accumulated_cost += err
    cost = accumulated_cost / (2 * m)
    return cost


def compute_gradient(x, y, w, b):
    m = len(x)
    n = len(x[0])
    grad_w = np.zeros_like(w)
    grad_b = 0

    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        err = f_wb_i - y[i]
        for j in range(n):
            grad_w[j] += err * x[i][j]
        grad_b += err
    grad_w = grad_w / m
    grad_b = grad_b / m

    return grad_w, grad_b


def gradient_descent(x, y, w_in, b_in, cost_function, alpha, num_iters, compute_gradient):
    cost_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        #for j in range(len(x[0])):
        #    tmp_w[j] = w[j] - alpha * dj_dw[j]
        # using vectorized operation helps code run more efficently 
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if num_iters < 100000:
            cost_history.append(cost_function(x,y,w,b))

        if i % 1000 == 0:
            cost = cost_function(x,y,w,b)
            print(f"The cost at iteration {i} is {cost}")

    print(f"Final cost: {cost_function(x,y,w,b)}")
    print(f"Final w: {w} Final b: {b}")
    return w, b, cost_history

def plot_cost_vs_iter(cost_history):
    fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    ax1.plot(cost_history)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Cost")
    ax1.set_title("Cost vs Iteration")

    cost_history_length = len(cost_history)
    desired = cost_history_length / 10
    desired = cost_history_length - desired

    final = int(desired)

    ax2.plot(final + np.arange(len(cost_history[final:])), cost_history[final:])
    ax2.set_xlabel("Iterations")
    ax1.set_ylabel("Cost")
    ax1.set_title("Cost vs Iteration (Tail)")
    plt.show()

def prediction_by_feature(x, y, w, b):
    m = x.shape[0]
    y_prediction = np.zeros(m)
    for i in range(m):
        y_prediction[i] = np.dot(x[i], w) + b
    
    fix, axs = plt.subplots(1, 5, figsize=(12,3), sharey=True)
    features = ["Sqft", "# bedrooms", "# bathrooms", "lot size", "age"]

    for i in range(5):
        axs[i].scatter(x[:,i], y, c='r', marker="x", label="Target")
        axs[i].scatter(x[:,i], y_prediction, c='b', label="Prediction")
        axs[i].set_xlabel(features[i])
        axs[i].set_title(f"{features[i]} vs price")
    axs[0].set_ylabel("Price")
    axs[0].legend()

    fix.tight_layout()
    plt.show()

def prediction(features, w, b, x):
    normalized_features = np.zeros_like(features)
    for i in range(x.shape[1]):
        min = np.min(x[:, i])
        max = np.max(x[:,i])
        normalized_features[i] = (features[i] - min) / (max - min)

    price = np.dot(normalized_features, w) + b
    print(f"The predicted price is ${price:.2f}")
    return price


file_path = "data.txt"
data = np.loadtxt(file_path)

x_train = data[:, :5]
y_train = data[:, 5]

    
def main():


    w = np.array([0,0,0,0,0])
    b = 0
    alpha = .3
    iters = 8000
    normalized_x_train = normalize_x_data(x_train)

    w, b, cost_history = gradient_descent(normalized_x_train, y_train, w, b, cost_function, alpha, iters, compute_gradient)

    plot_cost_vs_iter(cost_history)
    prediction_by_feature(normalized_x_train, y_train, w, b)
    prediction(np.array([2578, 4, 3, .22, 24]), w, b, x_train)

if __name__ == "__main__":
    main()