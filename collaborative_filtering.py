import numpy as np
import pandas as pd
import tensorflow as tf
import keras

def main():
    Y, R = load_movie_ratings_data()
    num_movies, num_users = Y.shape
    num_features = 100
    W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
    X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
    b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')
    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=1e-1)
    
    iterations = 200
    lambda_ = 1
    for iter in range(iterations):
        # Use TensorFlow’s GradientTape
        # to record the operations used to compute the cost 
        with tf.GradientTape() as tape:

            # Compute the cost (forward pass included in cost)
            cost_value = cofi_cost_func_v(X, W, b, Y, R, lambda_)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss
        grads = tape.gradient( cost_value, [X,W,b] )

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients( zip(grads, [X,W,b]) )

        # Log periodically.
        if iter % 20 == 0:
            print(f"Training loss at iteration {iter}: {cost_value:0.1f}")

 
def load_movie_ratings_data() -> tuple[np.ndarray, np.ndarray]:
    """loads movie ratings

    Returns:
        tuple[np.ndarray, np.ndarray]: 
        ratings from of 0.5 to 5 in 0.5 step increments; 0 means user didnt rate (n_movies, n_users), 
        array of 1 if rated 0 if not (n_movies, n_users)
    """
    file_path = "data/movie_ratings/ratings.csv"
    df = pd.read_csv(file_path)
    pivot_df = df.pivot(index="movieId", columns="userId", values="rating").fillna(0)
    user_ratings = pivot_df.to_numpy()
    R = (user_ratings != 0).astype(int)
    return user_ratings, R

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
        X (ndarray (num_movies,num_features)): matrix of item features
        W (ndarray (num_users,num_features)) : matrix of user parameters
        b (ndarray (1, num_users)            : vector of user parameters
        Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
        R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
        lambda_ (float): regularization parameter
    Returns:
        J (float) : Cost
    """
    pred = X @ W.T + b
    J = np.sum((R * (pred - Y)**2) / 2)

    reg_W = np.sum(W**2 * lambda_/2)
    reg_X = np.sum(X**2 * lambda_/2)   
    J += reg_W + reg_X        

    return J

if __name__ == "__main__":
    main()