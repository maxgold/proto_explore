from pykdtree.kdtree import KDTree




class KNN:
    def __init__(self, X_src, y_src, leaf_size=10):
        # self.knn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(X_src)
        self.kdtree = KDTree(X_src, leafsize=leaf_size)
        self.y_src = y_src

    def query_k(self, X, k):
        dist, indices = self.kdtree.query(X, k=k)
        return self.y_src[indices]
    def query_r(self, X, r):
        indices = self.kdtree.query_radius(X, r=r)
        return self.y_src[indices]







