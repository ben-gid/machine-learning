import math
from typing import Callable
import numpy as np

def mean_squared_error(y_hats: np.ndarray, ys: np.ndarray) -> np.float64:
    """_summary_

    Args:
        y_hats (np.ndarray): array of predictions
        ys (np.ndarray): array of targets

    Returns:
        np.float64: cost
    """
    m = y_hats.shape[0]
    return np.sum((y_hats -ys)**2) / (2 * m)

def compute_gradient_w(y_hats: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> float:
    """_summary_

    Args:
        y_hats (np.ndarray): array of predictions
        ys (np.ndarray): array of targets
        xs (np.ndarray): array of training data

    Returns:
        float: weight gradient
    """
    m = y_hats.shape[0]
    return xs.T @ (y_hats - ys) / m

def compute_gradient_b(y_hats: np.ndarray, ys: np.ndarray) -> float:
    """_summary_

    Args:
        y_hats (np.ndarray): array ofpredictions
        ys (np.ndarray): array of targets

    Returns:
        float: base gradient
    """
    m = y_hats.shape[0]
    return np.sum(y_hats - ys) / m


def predict(xs: np.ndarray, w: np.ndarray | float, b: float) -> np.ndarray:
    """_summary_

    Args:
        xs (np.ndarray): array of training data
        w (float | np.ndarray ): weight (float for single dimension training data, 
            array for > 1)
        b (float): base

    Returns:
        np.ndarray: prediction (y_hat)
    """
    return xs.dot(w) + b

def compute_gradients(
    xs: np.ndarray, 
    ys: np.ndarray, 
    w: float | np.ndarray, 
    b: float
) -> tuple[float, float]: 
    """_summary_

    Args:
        xs (np.ndarray): array of training data
        ys (np.ndarray): array of targets
        w (float | np.ndarray ): weight (float for single dimension training data, 
            array for > 1)
        b (float): base

    Returns:
        tuple[float, float]: gradients (weight gradient, base gradient)
    """
    y_hats = predict(xs, w, b)
    dj_dw = compute_gradient_w(y_hats, ys, xs)
    dj_db = compute_gradient_b(y_hats, ys)
    
    return dj_dw, dj_db
    
    
def gradient_descent(
    xs: np.ndarray, 
    ys: np.ndarray, 
    w_in: float | np.ndarray,
    b_in: float,
    alpha: float, 
    iterations: int,
    cost_function: Callable[[np.ndarray, np.ndarray], np.float64], 
    gradient_function: Callable[[np.ndarray, np.ndarray, float | np.ndarray, float], 
                                tuple[float, float]],
) -> tuple[float | np.ndarray, float, list[np.float64], list[list[float]]]:
    """_summary_

    Args:
        xs (np.ndarray): array of training data
        ys (np.ndarray): array of targets
        w_in w (float | np.ndarray ): starting weight (float for single dimension 
            training data, array for > 1)
        b_in (float): starting base
        alpha (float): learning rate
        iterations (int): number of iterations to calculate gradient and update parameters
        cost_function (Callable[[np.ndarray, np.ndarray], np.float64]): function to calculate cost 
            (mean_squared_error preferred to reach global minimun)
        gradient_function (Callable[[np.ndarray, np.ndarray, float, float], tuple[float, float]]): 
            function to calculate gradient

    Returns:
        tuple[float, float, list[np.float64], list[list[float]]]: [final weight, final base, 
        cost_history, param_history]
    """
    
    cost_history: list[np.float64] = []
    param_history: list[list[float]] = []
    w = w_in
    b = b_in    
    
    for i in range(iterations):
        # Calculate gradient
        dj_dw, dj_db = gradient_function(xs, ys, w, b)
        
        # Update parameters
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost = cost_function(predict(xs, w, b), ys)
            cost_history.append(cost)
            param_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        interval = max(1, math.ceil(iterations / 10))
        if i % interval == 0:
            print(f"Iteration {i:4}: Cost {cost_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
        
    return w, b, cost_history, param_history

def main():
    x_train = np.array([1.0, 2.0])   #features
    y_train = np.array([300.0, 500.0])   #target value
    # initialize parameters
    w_init = 0
    b_init = 0
    # some gradient descent settings
    iterations = 100000
    alpha = 1.0e-2
    # run gradient descent
    w_final, b_final, _, _ = gradient_descent(x_train ,y_train, w_init, b_init, alpha, 
                                                        iterations, mean_squared_error, compute_gradients)
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

def main2():
    x_train = np.array([1.0, 2.0])   #features
    y_train = np.array([300.0, 500.0])   #target value
    w = 0. # weight
    b = 0. # base
    alpha = 1.0e-2 # learning rate
    
    for _ in range(100_000):
        y_hats = predict(x_train, w, b)
        
        dj_dw, dj_db = compute_gradients(x_train, y_train, w, b)
        
        w -= alpha * dj_dw
        b -= alpha * dj_db
    print(f"(w,b) found by gradient descent: ({w:8.4f},{b:8.4f})")
    y_hats = predict(x_train, w, b)
    cost = mean_squared_error(y_hats, y_train)
    print(f"final_cost:{cost:.10f}")
    
    
if __name__ == "__main__":
    main()
    main2()