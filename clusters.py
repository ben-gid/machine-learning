import numpy as np
from typing import Optional

def main():
    rng = np.random.default_rng(seed=42)
    X = rng.integers(low= 1, high=10, size=(10, 2))
    best_centroids, lowest_cost, highest_cost = loop_cluster(X, 3, 100)
    print(best_centroids, lowest_cost, highest_cost)

def cluster_cost(X: np.ndarray, centroids: np.ndarray):
    loss = np.array([np.linalg.norm(X - centroid)**2 for centroid in centroids])
    return loss.mean(axis=0)

def cluster(X: np.ndarray, cluster_count: int, seed:Optional[int]=None) -> np.ndarray:
    """clusters dataset into specified clusters count

    Args:
        X (np.ndarray): data
        cluster_count (int): count to cluster data into
        seed (Optional[int], optional): seed to make rng ordered(not random). Defaults to None.

    Returns:
        np.ndarray: centroids of clusters
    """
    rng = np.random.default_rng(seed=seed)
    n_samples, n_features = X.shape
    indices = rng.choice(n_samples, size=cluster_count, replace=False)
    centroids = X[indices].copy()
    
    prev_asignments = np.zeros(n_samples)
    while True:
        # get the distance between each x and centroid
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
        # assign each x to the closest centroid
        assignments = np.argmin(distances, axis=0)
        # return if converged
        if np.array_equal(assignments, prev_asignments):
            return centroids
        # if not converged set centroids to mean of assigned xs
        new_centroids = []
        for i in range(cluster_count):
            points_in_cluster = X[assignments == i]
            if points_in_cluster.shape[0] > 0:
                new_centroids.append(points_in_cluster.mean(axis=0))
            else:
                new_centroids.append(centroids[i])
        centroids = np.array(new_centroids)
        prev_asignments = assignments
        
def loop_cluster(X: np.ndarray, cluster_count: int, loops:int):
    centroids = []
    for _ in range(loops):
        centroids.append(cluster(X, cluster_count))
    costs = [cluster_cost(X, centroids_) for centroids_ in centroids]
    lowest_cost = min(costs)
    highest_cost = max(costs)
    best_centroids = centroids[costs.index(lowest_cost)]
    return best_centroids, lowest_cost, highest_cost
    
if __name__ == "__main__":
    main()