import numpy as np
from generate_data import generate_separable

def main():
    X, y = generate_separable()
    alpha = 1e-3
    
    # -------- Non-regularized --------
    w_nonreg = np.zeros(X.shape[1])
    b_nonreg = 0.0

    for _ in range(100_000):
        dj_dw, dj_db = compute_sig_gradient(X, y, w_nonreg, b_nonreg)
        w_nonreg -= alpha * dj_dw
        b_nonreg -= alpha * dj_db

    final_cost = binary_cross_entropy(X, y, w_nonreg, b_nonreg)
    print(f"non regularized: {final_cost=}, {w_nonreg=}, {b_nonreg=}")

    # -------- Regularized --------
    w_reg = np.zeros(X.shape[1])
    b_reg = 0.0
    lamb = 1.0
    alpha = 1e-3

    for _ in range(100_000):
        dj_dw, dj_db = compute_regularized_sig_gradient(X, y, w_reg, b_reg, lamb)
        w_reg -= alpha * dj_dw
        b_reg -= alpha * dj_db

    final_cost = regularized_classification_cost(X, y, w_reg, b_reg, lamb)
    print(f"regularized: {final_cost=}, {w_reg=}, {b_reg=}")

    # -------- compare unregularized losses -------- 
    J_nonreg = binary_cross_entropy(X, y, w_nonreg, b_nonreg)
    J_reg_on_reg = binary_cross_entropy(X, y, w_reg, b_reg)

    print("unreg loss (nonreg weights):", J_nonreg)
    print("unreg loss (reg weights):", J_reg_on_reg)

def sigmoid(X: np.ndarray, w: float | np.ndarray, b: float):
    z = X.dot(w) + b
    return 1/ (1 + np.exp(-z))

def unsimplified_classification_cost(    
    xs: np.ndarray, 
    ys: np.ndarray, 
    w: float | np.ndarray, 
    b: float
) -> float: 
    losses = []
    for i in range(len(ys)):
        sig = sigmoid(xs[i], w, b)
        if ys[i] == 1:     
            loss = -np.log(sig)
        else: # ys[i] == 0
            loss = -np.log(1- sig)
        losses.append(loss)
    return sum(losses) / len(losses)

def binary_cross_entropy(
    xs: np.ndarray, 
    ys: np.ndarray, 
    w: float | np.ndarray, 
    b: float, 
    eps: float = 1e-12
) -> float: 
    y_hats = sigmoid(xs, w, b) # prediction
    y_hats = np.clip(y_hats, eps, 1 - eps) # to prevent log(0)
    cost = -(ys * np.log(y_hats) + (1 - ys) * np.log(1 - y_hats)).mean()  # log cost formula
        
    return cost 

def predict(X: np.ndarray, w: float | np.ndarray, b: float):
    proba = sigmoid(X, w, b)
    
    return (proba >= 0.5).astype(int)
    
def compute_sig_gradient(xs: np.ndarray, ys: np.ndarray,
                           w: float | np.ndarray, b: float) -> tuple[np.ndarray, float]:
    m = xs.shape[0]
    y_hats = sigmoid(xs, w, b)
    dj_dw = (xs.T @ (y_hats - ys)) / m
    dj_db = np.sum(y_hats - ys) / m
    
    return dj_dw, dj_db

# -------- Regularized Functions --------
def regularized_classification_cost(
    xs: np.ndarray, 
    ys: np.ndarray, 
    w: float | np.ndarray, 
    b: float, 
    lamb: float,
    eps: float = 1e-12
) -> float: 
    
    m = xs.shape[0]
    y_hats = sigmoid(xs, w, b) # prediction
    y_hats = np.clip(y_hats, eps, 1 - eps) # to prevent log(0)
    cost = - (ys * np.log(y_hats) + (1 - ys) * np.log(1 - y_hats)).mean() # log cost formula
    cost += (lamb/(2*m)) * np.sum(w**2) # add regularization  
    
    return cost

def compute_regularized_sig_gradient(
    xs: np.ndarray, 
    ys: np.ndarray,
    w: float | np.ndarray, 
    b: float, 
    lamb: float
) -> tuple[np.ndarray, float]:
    
    m = xs.shape[0]
    y_hats = sigmoid(xs, w, b)
    dj_dw = (xs.T @ (y_hats - ys)) / m
    dj_dw += (lamb/m) * w # add regularization gradient
    # no regularization for b
    dj_db = np.sum(y_hats - ys) / m
    
    return dj_dw, dj_db

    
if __name__ == "__main__":
    main()