import numpy as np
from sklearn.model_selection import train_test_split
from typing import Sequence, Optional
from numbers import Number
from generate_data import generate_separable

def main():
    X, y = generate_separable(20)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SigmoidNN()
    model.sequential(
        [
            DenseLayer(units=10),
            DenseLayer(units=20),
            DenseLayer(units=10),
            DenseLayer(units=1),
        ]
    )
    model.compile(features=X_train.shape[1], alpha=0.01)
    model.fit(X_train, y_train, epochs=10)
    predictions = model.predict(X_test)
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
    z = X.T @ W + b  
    return 1 / (1 + np.exp(-z))  

class DenseLayer:
    def __init__(self, units: int) -> None:
        self._W: Optional[np.ndarray] = None
        self.last_input = None
    
    @property
    def W(self) -> Optional[np.ndarray]:
        return self._W
    
    @W.setter
    def W(self, neurons: np.ndarray):
        self._W = neurons

class SigmoidNN:
    def __init__(self) -> None:
        pass
    
    def sequential(self, layers: list[DenseLayer]) -> None:
        pass
    
    def compile(self, features: int, alpha: float) -> None:
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    