import numpy as np
import matplotlib.pyplot as plt

class KMeans():
    def __init__(self):
        pass

    def pairwise_dist(self, x, y):
        # use euclidean distance to find dist where dist[i,j] = ||Xi - Yj||
        np.random.seed(1)
        dist = np.sqrt(np.sum((x[:, np.newaxis] - y[np.newaxis]) ** 2, axis=2))
        return dist

    def _init_centers(self, points, K, **kwargs):
        # random points as initialized centers
        np.random.seed(1)
        centroids = points.copy()
        np.random.shuffle(centroids)
        return centroids[:K]

    def _update_assignment(self, centers, points):
        # assign points to closest center
        np.random.seed(1)
        dist = self.pairwise_dist(centers, points)
        return np.argmin(dist, axis=0)

    def _update_centers(self, old_centers, cluster_idx, points):
        # update centers based on mean of clusters
        np.random.seed(1)
        return np.array([points[cluster_idx == k].mean(axis=0) for k in range(old_centers.shape[0])])

    def _get_loss(self, centers, cluster_idx, points):
        # loss function to minimize
        np.random.seed(1)
        loss = np.zeros(points.shape[0])
        for i in range(centers.shape[0]):
            loss[cluster_idx == i] = np.linalg.norm(points[cluster_idx == i] - centers[i], axis=1)
        return np.sum((loss) ** 2)

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        return cluster_idx, centers, loss

    def find_optimal_num_clusters(self, data, max_K=15):
        # see how Kmeans performs with different number of clusters (loss function by iteration)
        np.random.seed(1)
        X = []
        Y = []
        for i in range(max_K):
            cluuster_idx, centers, loss = self.__call__(data, i + 1)
            X = X + [i + 1]
            Y = Y + [loss]

        plt.plot(X, Y)
        return np.array(Y)