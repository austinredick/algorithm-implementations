class GMM():
    def __init__(self):  # No need to implement
        pass

    # helper functions
    def softmax(self, logits):
        e = np.exp(logits - np.max(logits, axis=1).reshape((-1, 1)))
        return e / e.sum(axis=1).reshape((-1, 1))

    def logsumexp(self, logits):
        e = np.exp(logits - np.max(logits, axis=1).reshape((-1, 1)))
        sumE = e.sum(axis=1).reshape((-1, 1))
        logSE = np.log(sumE)
        addBack = np.max(logits, axis=1)
        return logSE + addBack

    def _init_components(self, points, K, **kwargs):
        shape = points.shape
        n, d = shape

        # Initialize mixing coefficients pi
        pi = np.full(K, 1 / K)

        # random initialization for mu from dataset
        mu = points[np.random.randint(n, size=K)]

        # Initialize covariance
        sigma = np.array([np.eye(d) for i in range(K)])

        return pi, mu, sigma

    def _ll_joint(self, points, pi, mu, sigma):
        # assunes independence
        shape = points.shape
        n, d = shape
        k = len(pi)

        ll = np.zeros((n, k))
        for i in range(k):
            mu_i = points - mu[i, :]
            var = np.linalg.det(sigma[i, :, :])
            st_dev = np.sqrt(var)
            dn = np.sqrt((2 * np.pi) ** points.shape[1])
            ex = np.sum((-1 / 2) * np.multiply((mu_i @ np.linalg.inv(sigma[i, :, :])).T, mu_i.T), axis=0)
            pdf = (1.0 / ((st_dev * dn))) * np.exp(ex)
            logpi = np.log(pi[i] + 1e-32)
            lognorm = np.log(pdf + 1e-32)
            ll[:, i] = logpi + lognorm

        return ll

    def _E_step(self, points, pi, mu, sigma):
        ll = self._ll_joint(points, pi, mu, sigma)
        gamma = self.softmax(ll)

        return gamma

    def _M_step(self, points, gamma):
        # obtain the shapes
        shape1 = points.shape
        n, d = shape1
        shape2 = gamma.shape
        j, k = shape2

        # compute n_k
        n_k = np.sum(gamma, axis=0)

        pi = n_k / n
        mu = gamma.T.dot(points) / (n_k[:, np.newaxis])

        sigma = []
        for i in range(k):
            diff = points - mu[i]
            w_sum = np.diagonal(np.dot(gamma[:, i].T * diff.T, diff))
            sigma_i = w_sum / n_k[i]
            sigma_i = sigma_i * np.eye(d)
            sigma.append(sigma_i)
        sigma = np.array(sigma)

        return pi, mu, sigma

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16):

        pi, mu, sigma = self._init_components(points, K)
        for it in range(0, max_iters):
            # E-step
            gamma = self._E_step(points, pi, mu, sigma)

            # M-step
            pi, mu, sigma = self._M_step(points, gamma)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(points, pi, mu, sigma)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if it % 10 == 0:  print('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)