import numpy as np
import tensorflow as tf
import keras
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=1.5)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = keras.Sequential(
    [
        keras.layers.Dense(units= 10, activation="relu"),
        keras.layers.Dense(4, activation='linear') # logits
    ]
)

model.compile(
    optimizer='adam', 
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=100)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"{loss=}, {accuracy=}")



