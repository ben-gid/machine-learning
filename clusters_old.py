import numpy as np
from typing import Optional

def main():
    rng = np.random.default_rng(seed=42)
    X = rng.random(size=(10, 2))


def cluster_unvectorized(X:np.ndarray, clusters_count:int, seed:Optional[int]=None
            ) -> tuple[list[set], np.ndarray]:
    # TODO: use better random generator and vectorize
    rng = np.random.default_rng(seed=seed)
    examples, feutures = X.shape
    # set mius to random points
    mius = rng.random(size=(clusters_count, feutures))
    
    prev_clusters = None
    while True:
        # reset clusters after every loop
        clusters: list[set] = [set() for _ in range(clusters_count)]
        
        for i in range(examples):
            # get the distance for each miu to each x
            norms = [np.linalg.norm(X[i] - miu) for miu in mius]
            min_norm = min(norms)
            min_idx = norms.index(min_norm)
            # add index of x with the smallest norm to its cluster
            clusters[min_idx].add(i)
        mius = np.array([np.array([X[i] for i in c]).mean(axis=0) for c in clusters])
        # return if converged
        if prev_clusters == clusters:
            return clusters, mius
        # if not converged yet
        prev_clusters = clusters
        
def cluster(X: np.ndarray, clusters_count:int, seed: Optional[int]=None
            ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=seed)
    samples, feutures = X.shape
    # set mius to random indeces of X
    indices = rng.choice(samples, size=clusters_count, replace=False)
    centroids = X[indices].copy()
    
    prev_assignments = np.zeros(samples)
    
    while True:
        # get norms (norms will have shape of (clusters_count, X.shape[1]))
        # each row will have the norms of X - centroid
        # each column represents each centroid
        norms = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
        # assigned centroid index
        assignments = np.argmin(norms, axis=0)
        # return if converged
        if np.array_equal(assignments, prev_assignments):
            clusters = centroids[assignments]
            return clusters, centroids
        # TODO: vectorize
        new_centroids = []
        for i in range(clusters_count): 
            points_in_cluster = X[assignments == i]
            if len(points_in_cluster) > 0:
                new_centroids.append(points_in_cluster.mean(axis=0))
            else:
                # If a cluster is empty, keep it where it was 
                # TODO: or re-initialize it to a random point
                new_centroids.append(centroids[i])
        centroids = np.array(new_centroids)
        prev_assignments = assignments
                 
            
if __name__ == "__main__":
    main()