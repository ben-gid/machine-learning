import numpy as np

def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(z) / np.sum(z)