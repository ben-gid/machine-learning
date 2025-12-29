import numpy as np
import sklearn.model_selection as ms
from generate_data import generate_separable

try:
    import cupy as cp
    xp = cp
    DEVICE = "GPU (RTX 5070)"
except ImportError:
    xp = np
    DEVICE = "CPU"

def vec_sigmoid(X, W, b):
    z = X @ W + b 
    return 1 / (1 + xp.exp(-z))  

def binary_cross_entropy(a_out, y):
    epsilon = 1e-15
    a_out = xp.clip(a_out, epsilon, 1 - epsilon)
    return -(y * xp.log(a_out) + (1 - y) * xp.log(1 - a_out)).mean()

class DenseLayer:
    def __init__(self, units, input_features):
        self.W = xp.random.standard_normal(size=(input_features, units)) * 0.1
        self.b = xp.zeros(units)
    
    def backward(self, delta, A_in, m, alpha):
        dj_dw = (A_in.T @ delta) / m
        dj_db = xp.sum(delta, axis=0) / m
        self.W -= alpha * dj_dw
        self.b -= alpha * dj_db

class SigmoidNN:
    def __init__(self, alpha=0.1):
        self.layers = []
        self.alpha = alpha
    
    def fit(self, X, y, epochs=2000):
        for epoch in range(epochs):
            # Forward
            activations = [X]
            cur = X
            for layer in self.layers:
                cur = layer.forward(cur) if hasattr(layer, 'forward') else vec_sigmoid(cur, layer.W, layer.b)
                activations.append(cur)
            
            # Backward
            delta = activations[-1] - y
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]
                A_in = activations[i]
                if i > 0:
                    upstream = delta @ layer.W.T
                    sig_p = A_in * (1 - A_in)
                    layer.backward(delta, A_in, X.shape[0], self.alpha)
                    delta = upstream * sig_p
                else:
                    layer.backward(delta, A_in, X.shape[0], self.alpha)
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch} | Cost: {binary_cross_entropy(activations[-1], y):.4f}")

def main():
    print(f"Running on: {DEVICE}")
    
    # 1. Generate data on CPU
    X_raw, y_raw = generate_separable(10000) 
    
    # 2. Split on CPU (Scikit-learn safe)
    X_tr_cpu, X_te_cpu, y_tr_cpu, y_te_cpu = ms.train_test_split(X_raw, y_raw, test_size=0.2)
    
    # 3. Move training data to GPU
    X_train = xp.array(X_tr_cpu)
    y_train = xp.array(y_tr_cpu).reshape(-1, 1)
    
    model = SigmoidNN(alpha=0.5)
    model.layers = [
        DenseLayer(16, X_train.shape[1]),
        DenseLayer(8, 16),
        DenseLayer(1, 8)
    ]
    
    model.fit(X_train, y_train, epochs=3000)
    
    # 4. Predict and move back to CPU for reporting
    p_gpu = xp.array(X_te_cpu)
    for layer in model.layers:
        p_gpu = vec_sigmoid(p_gpu, layer.W, layer.b)
    
    # Move back to CPU (Standardize for accuracy check)
    if xp.__name__ == 'cupy':
        final_probs = cp.asnumpy(p_gpu)
    else:
        final_probs = p_gpu

    preds = (final_probs >= 0.5).astype(int).flatten()
    acc = (preds == y_te_cpu).mean()
    print(f"Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()