import numpy as np

def generate_separable(num_examples: int=200) -> tuple[np.ndarray, np.ndarray]:
    """generates seperable data for binary classification.

    Args:
        num_examples (int, optional): number of examples to generate. Defaults to 200.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
        X: 2d array of examples size=(num_examples, 2)
        y: targets, 1d array of 0 or 1
    """
    X = np.random.uniform(-3, 3, size=(num_examples, 2))
    y = (X[:, 0] > X[:, 1]).astype(int)
    return X, y

def generate_separable_1d(num_examples: int=10):
    sep_idx = np.random.randint(int(num_examples / 2)) + num_examples / 4
    X = np.arange(num_examples)
    y = (X >= sep_idx).astype(int)
    return X, y
    