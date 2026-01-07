import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    X, y = fetch_openml(name="house_prices", as_frame=True, return_X_y=True)
    print(X.data)

if __name__ == "__main__":
    main()