import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.activations import sigmoid

from generate_data import generate_separable
from plot import plot_binary_classification

X, y = generate_separable(500)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state= 25, shuffle=True)

plot_binary_classification(X, y)
plt.savefig("imgs/neurons_data.png")

X_train = np.tile(X_train, (1000, 1))
y_train = np.tile(y_train, (1000))
print(f"{X_train.shape=}, {y_train.shape=}")

model = keras.Sequential(
    [
        keras.layers.Dense(units=3, activation=sigmoid, name="L1"),
        keras.layers.Dense(units=6, activation=sigmoid, name="L2"),
        keras.layers.Dense(units=3, activation=sigmoid, name="L3"),
        keras.layers.Dense(units=1, activation=sigmoid, name="L4"),
    ]
)
model.compile(loss="binary_crossentropy")
model.fit(X_train,y_train, epochs=5)
prob = model.predict(X_test)
accuracy = prob[np.where(prob != y_test)].mean()
evl = model.evaluate(X_test, y_test)
print(f"{evl=}, {accuracy=}")
