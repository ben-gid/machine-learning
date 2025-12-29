import numpy as np
from sklearn.model_selection import train_test_split
from generate_data import generate_separable

def main():
    X, y = generate_separable(100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    features = X.shape[1]
    model = SigmoidNN()
    model.sequential(
        [
            DenseLayer(units=10, input_features=features),
            DenseLayer(units=100, input_features=10),
            DenseLayer(units=50, input_features=100),
            DenseLayer(units=1, input_features=50),
        ]
    )
    X_train = np.tile(X_train, (1_000, 1))
    y_train = np.tile(y_train, (1_000))
    
    model.compile(alpha=0.1)
    model.fit(X_train, y_train, epochs=20)
    predictions = model.predict(X_test)
    pred_classes = (predictions >= 0.5).astype(int)
    accuracy = (pred_classes == y_test.reshape(-1, 1)).mean()
    print(f"{accuracy=}")
    # print(f"final cost on test set: {final_cost}")
       
def vec_sigmoid(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """vecorized sigmoid

    Args:
        X (np.ndarray): 2d array of data
        W (np.ndarray): 2d array of weigts
        b (np.ndarray): 1d array of bias's

    Returns:
        np.ndarray: sigmoid
    """
    z = X @ W + b # each row of W is a feature to allow X @ W 
    return 1 / (1 + np.exp(-z))  

def binary_cross_entropy(a_out: np.ndarray, y: np.ndarray) -> float:
    """sigmoid cost function

    Args:
        a_out (np.ndarray): 1d array of output from node
        y (np.ndarray): 1d target values

    Returns:
        float: cost
    """
    return -(y * np.log(a_out) + (1 - y) * np.log(1 - a_out)).mean()

class DenseLayer:
    def __init__(self, units: int, input_features: int) -> None:
        # each column in w is a neuron and each row is a feature
        rng = np.random.default_rng()
        self._W: np.ndarray = rng.standard_normal(size=(input_features, units)) * 0.1
        self._b: np.ndarray = np.zeros(units)
        self.last_input = None
    
    @property
    def W(self) -> np.ndarray:
        return self._W
    
    @W.setter
    def W(self, W: np.ndarray):
        self._W = W
        
    @property
    def b(self) -> np.ndarray:
        return self._b
    
    @b.setter
    def b(self, b: np.ndarray):
        self._b = b
        
    def forward(self, A_in: np.ndarray) -> np.ndarray:
        return vec_sigmoid(A_in, self.W, self.b)
    
    def backward(self, delta:np.ndarray, A_in:np.ndarray, 
                 batch_size: int, alpha: float) -> None:
        
        dj_dw = (A_in.T @ delta) / batch_size
        dj_db = np.sum(delta, axis=0) / batch_size

        self.W -= alpha * dj_dw
        self.b -= alpha * dj_db

class SigmoidNN:
    def __init__(self) -> None:
        self.layers: list[DenseLayer] = []
        self.alpha: float = 1e-2
    
    def sequential(self, layers: list[DenseLayer]) -> None:
        self.layers = layers
    
    def compile(self, alpha: float) -> None:
        self.alpha = alpha
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        y = y.reshape(-1, 1)
        for epoch in range(epochs):
            # -------- Forwardprop --------
            activations = [X] # store activations for backprop
            current_A = X
            for layer in self.layers:
                current_A = layer.forward(current_A)
                
                activations.append(current_A)
                
            cost = binary_cross_entropy(current_A, y)
            # -------- Backprop --------
            delta: np.ndarray = activations[-1] - y
            print(f"cost is {cost:.2f} at epoch {epoch + 1}")
            for i in reversed(range(len(self.layers))):
                current_layer = self.layers[i]
                m = y.shape[0]
                A_in = activations[i]
                # calculate upstreem error before running backward, which updates the weights
                if i > 0:
                    upstream_error = delta @ current_layer.W.T
                    sig_prime = A_in * (1 - A_in)
                    current_layer.backward(delta, A_in, m, self.alpha)
                    # delta for next layer
                    delta = upstream_error * sig_prime
                else:
                    current_layer.backward(delta, A_in, m, self.alpha)
                         
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        current_A = X
        for layer in self.layers:
            current_A = layer.forward(current_A)
        return current_A
    
if __name__ == "__main__":
    main()