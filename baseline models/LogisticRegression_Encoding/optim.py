import numpy as np


class AdamOptimizer:
    """
    Minimal Adam optimizer with a `step_and_cost` API, mirroring
    PennyLane's qml.AdamOptimizer so the training loop from the original
    project didn't need to change shape -- just the model being optimized.

    theta_new = theta - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, stepsize=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step_and_cost(self, cost_fn, weights, grad_fn):
        """
        cost_fn(weights) -> scalar loss
        grad_fn(weights)  -> gradient array, same shape as weights
        """
        loss = cost_fn(weights)
        grad = grad_fn(weights)

        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        new_weights = weights - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return new_weights, loss
