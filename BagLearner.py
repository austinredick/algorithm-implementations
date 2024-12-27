import numpy as np

class BagLearner(object):
    def __init__(self, learner, kwargs = None, bags = 20, boost = False, verbose = False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []
        for i in range(self.bags):
            self.learners.append(self.learner(**self.kwargs))

    def add_evidence(self, x_train, y_train):
        for i, learner in enumerate(self.learners):
            # select random x to train on
            selection = np.random.choice(x_train.shape[0], x_train.shape[0], replace = True)
            x_train_bag = x_train[selection]
            y_train_bag = y_train[selection]
            # feed that into the learner
            learner.add_evidence(x_train_bag, y_train_bag)
            if self.verbose:
                print(f"Created model No. {i}")

    def query(self, x_test):
        predictions = []
        for i, learner in enumerate(self.learners):
            prediction = learner.query(x_test)
            predictions.append(prediction)
        return np.apply_along_axis(self.mode, axis=0, arr=predictions)

    def mode(self, predictions):
        # using mode for classification, mean will make more sense for other applications
        values, counts = np.unique(predictions, return_counts=True)
        idx = np.argmax(counts)
        return values[idx]

