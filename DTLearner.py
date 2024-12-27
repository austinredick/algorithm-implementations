import numpy as np

class DTLearner(object):
    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def add_evidence(self, x_train, y_train):
        # add training data and build tree
        self.tree = self.build_tree(x_train, y_train)
        if self.verbose:
            print(f"Tree shape after building: {self.tree.shape}")

    def build_tree(self, x, y):
        # if we reach leaf size or if all elements are identical --> leaf node (base cases)
        if len(set(y)) == 1 or x.shape[0] <= self.leaf_size:
            return np.array([-1, np.mean(y), np.nan, np.nan]).reshape(1, -1)
        # determine feature to split on via highest correlation
        corr = np.zeros(x.shape[1])  # Initialize a NumPy array for correlation values
        for i in range(x.shape[1]):
            if np.std(x[:, i]) == 0:  # Check for zero variance
                corr[i] = 0  # Assign a correlation of 0 for constant features
            else:
                corr[i] = abs(np.corrcoef(x[:, i], y)[0, 1])
        feat = np.argmax(corr)
        split_val = np.median(x[:,feat])
        # check for bad split
        if (np.sum(x[:,feat]<=split_val) == x.shape[0]) or (np.sum(x[:, feat] > split_val) == x.shape[0]):
            return np.array([-1, np.mean(y), np.nan, np.nan]).reshape(1,-1)
        # split based on selected feature + recursive call
        left = self.build_tree(x[x[:,feat]<=split_val], y[x[:,feat]<=split_val])
        right = self.build_tree(x[x[:, feat]>split_val], y[x[:, feat]>split_val])
        root = np.array([feat, split_val, 1, left.shape[0]+1]).reshape(1,-1)
        return np.concatenate([root, left, right], axis=0)

    def traverse(self, x, node_idx):
        # prediction for a single x value by moving through decision tree
        node = self.tree[node_idx]
        if node[0] == -1: #if it is leaf node return np.mean(y)
            return node[1]
        feat, split_val = int(node[0]), node[1]
        # Ensure the next node is within bounds before traversing
        if x[feat] <= split_val:
            next_node_idx = node_idx + int(node[2])
        else:
            next_node_idx = node_idx + int(node[3])
        if next_node_idx >= self.tree.shape[0]:
            return node[1]
        if x[feat] <= split_val: #if less than split val, go left, else go right
            return self.traverse(x, int(node_idx + node[2]))
        else:
            return self.traverse(x, int(node_idx + node[3]))

    def query(self, x_test):
        # predict for each value in our test data
        predictions = np.array([self.traverse(x,0) for x in x_test])
        return np.array(predictions)