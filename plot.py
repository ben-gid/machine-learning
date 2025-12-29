import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Any

def quadratic(x:np.ndarray, a: float, b:float, c: float)-> np.ndarray:
    """Computes a quadratic to a np.ndarray

    Args:
        x (np.ndarray): array of values to apply quadratic to 
        a (float): first scalar in quadratic
        b (float): second scalar in quadratic
        c (float): third scalar in quadratic

    Returns:
        np.ndarray: result of apply quadratic to x
    """
    y = a*x**2 + b*x + c
    return y     
    
def vertex_2(x: np.ndarray)-> tuple[np.ndarray, float, float]:
    # maximum (vertex) should be at x = 2
    a, b, c = -1, 4, 3
    y = quadratic(x, a, b, c)
    x_vertex, y_vertex = compute_vertex(a, b, c)
    return y, x_vertex, y_vertex


def compute_vertex(a: float, b: float, c: float) -> tuple[float, float]:
    """Computes vertex (x, y) for y = ax^2 + bx + c"""
    x_vertex = -b / (2*a)
    y_vertex = a * x_vertex**2 + b * x_vertex + c
    return x_vertex, y_vertex

def plot_function(
    x: np.ndarray, 
    func: Callable[[np.ndarray], np.ndarray]):
    """plots a function with matplotlip and saves it as the functions name .png

    Args:
        x (np.ndarray): array to apply function to
        func (Callable[[np.ndarray], float]): function to plot
    """
    _, ax = plt.subplots()
    y = func(x)
    ax.plot(x, y)
    func_name = func.__name__
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"function: {func_name}")

    plt.savefig(fname=f"imgs/{func_name}.png")
    
    
def plot_binary_classification(
    X: np.ndarray, 
    y: np.ndarray,
    x_names: list[str] | None= None
) -> None:
    """creates a matplotlib plot that plots x and y. you can view the plot after calling this fuction
    in jupyter with plt.show() and save it with plt.savefig.

    Args:
        X (np.ndarray): data training set
        y (np.ndarray): data target
        x_names (list[str] | None, optional): names for each x feature. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    
    features = X.shape[1]
    
    if features != 2:
        raise ValueError("X can only have 2 features")
    
    if x_names is None:
        x_names = [f"Feature {i}" for i in range(features)]
    
    if len(x_names) != 2:
        raise ValueError("x_names must contain exactly two labels.")

        
    plt.scatter(X[y==0, 0], X[y==0, 1], c="r", marker="o", label="class 0")
    plt.scatter(X[y==1, 0], X[y==1, 1], c ="b", marker="o", label="class 1")
    plt.xlabel(x_names[0])
    plt.ylabel(x_names[1])
    plt.title("Binary classification")
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_one_variable_classification(X: np.ndarray, Y: np.ndarray):   
    
    pos = Y == 1
    neg = Y == 0

    plt.scatter(X[pos], Y[pos], marker='x', s=80, c = 'red', label="y=1")
    plt.scatter(X[neg], Y[neg], marker='o', s=100, c= 'blue', label="y=0")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('one variable plot')
    plt.legend()
    
def plot_actual_vs_prediction(
    data: np.ndarray, 
    Y: np.ndarray, 
    y_hats: np.ndarray, 
    feature_names: Optional[list[Any]]=None,
) -> None:
    # plt.style.use("_mpl-gallery")
    
    if data.ndim > 1:
        features = data.shape[1]
    
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(features)]
            
        for i in range(features):
            plt.figure(figsize=(7,5))
            
            plt.scatter(data[:, i], Y, label="Y (true), alpha=0.7")
            plt.scatter(data[:, i], y_hats, label="Y Hat (predicted)", alpha=0.7)

            plt.xlabel(feature_names[i])
            plt.ylabel("Target")
            plt.title(f"Model Predictions vs True Values ({feature_names[i]})")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.3)
            
            plt.savefig(fname=f"performance{i}.png")
    
    else:
        if feature_names is None:
            feature_name = "Feature 1"
        else:
            feature_name = feature_names[0]   
        plt.figure(figsize=(7,5))
        
        plt.scatter(data, Y, label="Y (true), alpha=0.7")
        plt.scatter(data, y_hats, label="Y Hat (predicted)", alpha=0.7)

        plt.xlabel(feature_name)
        plt.ylabel("Target")
        plt.title(f"Model Predictions vs True Values ({feature_name})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        
        plt.savefig(fname="performance.png")
        
def plot_decision_boundary(X, y, w, b):
    # Scatter original data
    plt.scatter(X[y==0, 0], X[y==0, 1], label="Class 0")
    plt.scatter(X[y==1, 0], X[y==1, 1], label="Class 1")

    # Decision boundary line: w1*x1 + w2*x2 + b = 0
    x1_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    x2_vals = -(w[0] * x1_vals + b) / w[1]

    plt.plot(x1_vals, x2_vals, 'k-', label="Decision boundary")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
