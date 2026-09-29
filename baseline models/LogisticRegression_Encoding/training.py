import numpy as np
from logistic_model import logistic_model_batch
from optim import AdamOptimizer
from config import LEARNING_RATE, EPOCHS, BATCH_SIZE

EPS = 1e-12  # avoids log(0)


def binary_cross_entropy_loss(weights, X_batch, y_batch):  # loss
    """Computes binary cross-entropy loss over a batch of inputs."""
    probs = logistic_model_batch(weights, X_batch)  # fwd pass for the whole batch at once
    probs = np.clip(probs, EPS, 1 - EPS)
    return -np.mean(
        y_batch * np.log(probs) + (1 - y_batch) * np.log(1 - probs)
    )


def bce_gradient(weights, X_batch, y_batch):
    """Analytic gradient of the BCE loss w.r.t. weights (w) and bias (b).

    Conveniently, for logistic regression + BCE the gradient has the same
    clean form as linear regression + MSE: d(loss)/d(z) = (p - y).
    """
    n = len(X_batch)
    probs = logistic_model_batch(weights, X_batch)
    error = probs - y_batch  # d(loss)/d(z), where z = w.x + b

    grad_w = (1.0 / n) * (X_batch.T @ error)
    grad_b = (1.0 / n) * np.sum(error)

    return np.concatenate([grad_w, [grad_b]])


def calculate_accuracy(weights, X, y):
    """Computes binary classification accuracy (threshold probability at 0.5)."""
    probs = logistic_model_batch(weights, X)  # run the whole dataset
    binary_preds = (probs >= 0.5).astype(float)  # classify our output into 0/1 classes
    return np.mean(binary_preds == y)  # find the mean to get the accuracy


def train(weights, X, y):  # training fun that receives weights (current model parameters)
    """Optimization execution loop with Adam.
    theta_new = theta - lr * update
    calculate how each param affects loss, move param in the dir that reduces loss
    """
    opt = AdamOptimizer(stepsize=LEARNING_RATE)  # adaptive moment estimation that uses gradients to decide how much to update the parameters
    loss_history = []

    num_samples = len(X)

    for epoch in range(EPOCHS):  # complete pass through the training dataset.
        # Shuffle batch indices to prevent model from learning based on the order of examples
        indices = np.random.permutation(num_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]

        # mini batches with each batch produces one update
        for i in range(0, num_samples, BATCH_SIZE):
            X_batch = X_shuffled[i:i + BATCH_SIZE]
            y_batch = y_shuffled[i:i + BATCH_SIZE]
            # current weights -> calc gradient -> update weights using Adam optimizer
            # -> after each epoch evaluate loss of improved model
            weights, loss = opt.step_and_cost(
                lambda w: binary_cross_entropy_loss(w, X_batch, y_batch),
                weights,
                lambda w: bce_gradient(w, X_batch, y_batch)
            )

        # Record loss over full dataset for the epoch
        epoch_loss = binary_cross_entropy_loss(weights, X, y)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = calculate_accuracy(weights, X, y)
            print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {acc * 100:.1f}%")

    return weights, loss_history
