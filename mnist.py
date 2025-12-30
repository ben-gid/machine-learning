import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

# from https://keras.io/api/datasets/mnist/
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

model = keras.models.Sequential(
    [
        keras.layers.Input(shape=tuple(x_train.shape[1:])),
        keras.layers.Rescaling(1./255),
        keras.layers.Flatten(),
        keras.layers.Dense(units=25, activation='relu'),
        keras.layers.Dense(units=15, activation='relu'),
        keras.layers.Dense(units=10, activation='linear'),
    ]
)

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=100)

# accuracy calc 1
loss, accuracy = model.evaluate(x_test, y_test, verbose="0")
print(f"{loss=},{accuracy=}")

# accuracy calc 2
logits = model(x_test)
probs = tf.nn.softmax(logits)
preds = np.argmax(probs, axis=1)
y_true = y_test.reshape(-1)
y_pred = preds.reshape(-1)
accuracy = (y_true == y_pred).mean()
print(f"{accuracy=}")