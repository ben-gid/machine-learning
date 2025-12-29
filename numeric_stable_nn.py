"""Uses logits with fused binary cross-entropy to improve numerical stability 
and gradient reliability during training."""
import numpy as np
import tensorflow as tf
import keras
from keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split

from generate_data import generate_separable

def main():
    X, y = generate_separable(500)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
    
    model = keras.Sequential(
        [
            keras.layers.Dense(units=30, activation="relu", input_shape=(X.shape[1],)),
            keras.layers.Dense(units=20, activation="relu", input_shape=(X.shape[1],)),
            keras.layers.Dense(units=15, activation="relu", input_shape=(X.shape[1],)),
            # instead of sigmoid use linear for the last layer
            keras.layers.Dense(units=1, activation="linear")
        ]
    )
    # use this loss function to make calculating loss more accurate
    model.compile(
        loss=BinaryCrossentropy(from_logits=True), 
        optimizer="adam", 
        metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=100)
    
    # evaluate data using sigmoid
    logits = model(X_test)
    probs = tf.nn.sigmoid(logits)
    preds = (probs > 0.5).numpy().astype(int)
    
    y_true = y_test.reshape(-1)
    y_pred = preds.reshape(-1)
    accuracy = (y_true == y_pred).mean()
    print(f"{accuracy=}")
    
if __name__ == "__main__":
    main()