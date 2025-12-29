import numpy as np
from sklearn.model_selection import train_test_split
from typing import Sequence, Optional
from numbers import Number
from logistic_regression import sigmoid, compute_sig_gradient, predict
from generate_data import generate_separable

def main():
    X, y = generate_separable(20)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SigmoidNN()
    model.sequential(
        [
            DenseLayer(neuron_count=10),
            DenseLayer(neuron_count=20),
            DenseLayer(neuron_count=10),
            DenseLayer(neuron_count=1),
        ]
    )
    model.compile(features=X_train.shape[1], alpha=0.01)
    model.fit(X_train, y_train, epochs=10)
    predictions = model.predict(X_test)
    # print(f"final cost on test set: {final_cost}")
       
def binary_cross_entropy(a_out: np.ndarray, y: np.ndarray) -> float:
    """sigmoid cost function

    Args:
        a_out (np.ndarray): 1d array of output from node
        y (np.ndarray): 1d target values

    Returns:
        float: cost
    """
    return -(y * np.log(a_out) + (1 - y) * np.log(1 - a_out)).mean()
    
class Neuron:
    """neuron that uses the sigmoid function for learning

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # creates a neuron
    # each neuron contains w and b
    # each neuron takes in gradient and updates w and b with it
    # each neuron outputs a new prediction 
    def __init__(self, features: int) -> None:
        # initialize wwights randomly to break symmetry
        self._w = self._w = np.random.randn(features) * 0.01
        self._b = 0
    
    @property
    def w(self):
        return self._w
    
    @property
    def b(self):
        return self._b
    
    @w.setter
    def w(self, w:np.ndarray):
        if w.ndim != 1:
            raise ValueError("w can only be 1d")
        self._w = w
    
    @b.setter
    def b(self, b: float):
        if not isinstance(b, Number):
            raise ValueError("b can only be a float")
        self._b = b
    
    def a_out(self, X: np.ndarray):
        return self.w.dot(X) + self.b
      
    def forward(self, a_in: np.ndarray) -> np.ndarray:
        """predicts output using sigmoid

        Args:
            a_in (np.ndarray): activation; input as 2d array

        Returns:
            np.ndarray: output (y_hat)
        """
        return sigmoid(a_in, self.w, self.b)
    
    
    def sig_gradient(self, X:np.ndarray, y: np.ndarray):
        return compute_sig_gradient(X, y, self.w, self.b)
    
    def loss(self, a_out:np.ndarray, y:np.ndarray) -> float:
        return binary_cross_entropy(a_out, y)
    
class DenseLayer:
    # each layer containes multiple neurons
    # a layer should take in a list of data, send them through each neuron 
    # and return their output
    def __init__(self, neuron_count: int) -> None:
        self._neuron_count = neuron_count
        self._neurons = []
        self.last_input = None

    @property
    def neuron_count(self):
        return self._neuron_count
    
    @neuron_count.setter
    def neuron_count(self, count):
        self._neuron_count = count
    
    @property
    def neurons(self) -> Sequence[Neuron]:
        return self._neurons
    
    @neurons.setter
    def neurons(self, neurons: Sequence[Neuron]):
        ittr = isinstance(neurons, Sequence)
        neur = all(isinstance(n, Neuron) for n in neurons)
        if ittr is not True or neur is not True:
            raise ValueError("neurons can must be of type Sequence[Neuron]")
        
        self._neurons = neurons
        
    def compile(self, features: int) -> None:
        """adds all the neurons with weights set to 0 to the layer

        Args:
            features (int): features for w (X.shape[1])
        """
        self.neurons = [Neuron(features) for _ in range(self.neuron_count)]
    
    def proba(self, a_in:np.ndarray) -> np.ndarray: 
        """calculates the probability of all neurons

        Args:
            a_in (np.ndarray): input activation as 2d array to be passed to every neuron

        Returns:
            np.ndarray: 1d array of neurons proba
        """
        # transpose to get correct shape
        return np.array([neuron.forward(a_in) for neuron in self.neurons]).T 
                
    def forward(self, a_in: np.ndarray) -> np.ndarray:
        """Saves X to last_input and returns proba/sigmoid for forward propogation

        Args:
            a_in (np.ndarray): input activation as 2d array

        Returns:
            np.ndarray: probility of X; y_hat
        """
        self.last_input = a_in
        return self.proba(a_in)
    
    def backward(self, upstream_grad: np.ndarray, alpha: float) -> np.ndarray:
        """performs backpropagation for the layer

        Args:
            dA (np.ndarray): gradient of the cost with respect to the activation
            alpha (float): learning rate

        Returns:
            np.ndarray: gradient of the cost with respect to the weight
        """
        if not self.last_input:
            raise RuntimeError()
        raise NotImplementedError()

    
    def losses(self, X:np.ndarray, y:np.ndarray) -> np.ndarray:
        return np.array([n.loss(X, y) for n in self.neurons])
    
    def sig_gradients(self, a_in: np.ndarray, y:np.ndarray) -> list[tuple[np.ndarray, float]]:
        return [neuron.sig_gradient(a_in, y) for neuron in self.neurons]
    
    def weights(self) -> np.ndarray:
        return np.array([neuron.w for neuron in self.neurons])

class SigmoidNN:
    # creates a nerual network comprized of many layers.
    # containes multiple layers
    # function to train predict, and evaluate
    def __init__(self) -> None:
        self.layers = []
        self.alpha = None
        self.W = None
        
    def sequential(self, layers: list[DenseLayer]):
        self.layers = layers
        
    def compile(self, features: int, alpha=0.01):
        self.alpha = alpha
        features_current = features
        for layer in self.layers:
            layer.compile(features_current)
            features_current = layer.neuron_count

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int=1):
        if self.alpha is None:
            raise ValueError("you must run compile before fit")
        
        for _ in range(epochs):
            activations = [X] # store all activations for backprop
            current_a = X # set initial a to X
            for layer in self.layers:
                current_a = layer.forward(current_a)
                activations.append(current_a)
            
            # Assume 'activations' list contains: [X, output_layer1, output_layer2, ...]
            # activations[0] is X (Input)
            # activations[-1] is the Final Prediction

            # 1. Initialize the first 'delta' (error signal) at the Output Layer
            # We use (Prediction - y) because it is the derivative of Binary Cross Entropy + Sigmoid
            # Note: activations[-1] is the final prediction
            delta = activations[-1] - y 

            # 2. Iterate BACKWARDS
            # We start at the last layer index and go down to 0
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]
                
                # 'a_prev' is the input that was fed INTO this layer
                # If i is 0 (first layer), a_prev is X (activations[0])
                # because len(activations) == len(layers) + 1
                a_prev = activations[i] 
                
                # --- A. Calculate Gradients for THIS layer ---
                # dW = (Input_Transposed @ Delta) / Batch_Size
                m = a_prev.shape[0] # number of examples
                dw = (1 / m) * np.dot(a_prev.T, delta)
                db = (1 / m) * np.sum(delta, axis=0)
                
                # --- B. Calculate 'delta' for the NEXT loop iteration (the layer to the left) ---
                # We only need to do this if we are NOT at the first layer
                if i > 0:
                    # 1. Pull error back through weights: (Delta @ Weights_Transposed)
                    error_prop = np.dot(delta, layer.w.T)
                    
                    # 2. Multiply by derivative of sigmoid from previous layer
                    # sigmoid_derivative = a * (1 - a)
                    # We look at activations[i] because that represents the output of the previous layer relative to the next step
                    sig_deriv = a_prev * (1 - a_prev)
                    
                    # 3. New delta for the next step
                    delta = error_prop * sig_deriv
                    
                # --- C. Update Weights ---
                # (In a real library, you usually store gradients and update later, but this works)
                layer.w -= self.alpha * dw
                layer.b -= self.alpha * db
                
    def predict(self, X:np.ndarray) -> np.ndarray:
        a = X # set initial a to X
        
        for i in range(len(self.layers)):
            layer = self.layers[i] 
            a = layer.proba(a)
        return a
    
if __name__ == "__main__":
    main()