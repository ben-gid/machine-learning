import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

# supress regression warning
import warnings

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50%"
)

# 1. Setup Data
RS = 42
X, y = fetch_openml(name="house_prices", as_frame=True, return_X_y=True)
y = np.asarray(y, dtype=float)

# Select columns
numeric_cols = X.select_dtypes(include=["number"]).columns
object_cols = X.select_dtypes(include=["object"]).columns

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=RS)

# 2. Improved Pipelines
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    # Scaling is not strictly necessary for Random Forest, so we skip it
])

# Switch to OrdinalEncoder for tree-based models to handle high cardinality better
object_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("obj", object_pipeline, object_cols),
    ],
    remainder="drop"
)

# 3. Model Definition
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=RS, n_jobs=-1, oob_score=True))
])

# 4. Define Search Space (The "Grid")
# Instead of single loops, we define ranges for everything at once
param_distributions = {
    # Number of trees (more is usually better, but slower)
    "model__n_estimators": randint(100, 500),
    # Maximum depth of the tree (controls overfitting)
    "model__max_depth": randint(5, 30),
    # Minimum samples required to split a node (controls overfitting)
    "model__min_samples_split": randint(2, 20),
    # Minimum samples at a leaf node (strong control for variance)
    "model__min_samples_leaf": randint(1, 10),
    # Number of features to consider at every split
    "model__max_features": [1.0, "sqrt", "log2"] 
}

# 5. Run Optimization
print("Starting Hyperparameter Tuning...")
search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_distributions,
    n_iter=20,  # How many random combinations to try
    cv=3,       # 3-fold Cross Validation
    scoring="neg_root_mean_squared_error",
    random_state=RS,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

# 6. Evaluation
print(f"Best Params: {search.best_params_}")

best_model = search.best_estimator_
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)

print(f"R2 Train Score: {train_score:.4f}")
print(f"R2 Test Score:  {test_score:.4f}")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor

# 1. We keep your existing preprocessor (or simplify it)
# HistGradientBoosting handles missing values natively!
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols), # No imputer needed for HGBR
        ("obj", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), object_cols),
    ]
)

# 2. Define the Gradient Boosting Model
hgb = HistGradientBoostingRegressor(
    random_state=RS,
    max_iter=500,        # Equivalent to n_estimators
    learning_rate=0.05,  # Smaller steps are usually more accurate
    max_depth=10,
    l2_regularization=1.5
)

# 3. Wrap it in a Target Transformer
# This logs the price before training and "un-logs" it for predictions automatically
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", TransformedTargetRegressor(
        regressor=hgb, 
        func=np.log1p, 
        inverse_func=np.expm1
    ))
])

# 4. Fit and Score
model.fit(X_train, y_train)
print(f"New Train Score: {model.score(X_train, y_train):.4f}")
print(f"New Test Score:  {model.score(X_test, y_test):.4f}")

from xgboost import XGBRegressor

# 1. Update the pipeline to use XGBoost
# We add high 'reg_lambda' and 'reg_alpha' to force the model to be simpler
xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,             # Lower depth = less overfitting
    reg_lambda=10,           # L2 Regularization (Higher = more conservative)
    reg_alpha=1,             # L1 Regularization
    random_state=RS,
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", TransformedTargetRegressor(
        regressor=xgb_model, 
        func=np.log1p, 
        inverse_func=np.expm1
    ))
])

# 2. Fit the model
model.fit(X_train, y_train)

print(f"XGB Train Score: {model.score(X_train, y_train):.4f}")
print(f"XGB Test Score:  {model.score(X_test, y_test):.4f}")

def add_features(df):
    df = df.copy()
    # 1. Total Square Footage (often the #1 predictor)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # 2. Total Bathrooms
    df['TotalBath'] = (df['FullBath'] + (0.5 * df['HalfBath']) + 
                       df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
    
    # 3. Age of the house when sold
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    return df

# Apply it
X_custom = add_features(X)
X_train, X_test, y_train, y_test = train_test_split(X_custom, y, train_size=0.75, random_state=RS)