import numpy as np
from generate_data import generate_separable
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier


model =XGBClassifier()

X, y = generate_separable()

X_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.9, random_state=42, shuffle=True)
model.fit(X_train, y_train)
y_pred = model.predict(x_test)
print(y_test[np.where(y_test != y_pred)].shape[0] / y_test.shape[0])