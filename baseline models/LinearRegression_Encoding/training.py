import numpy as np
from linear_model import linear_model_batch
from optim import AdamOptimizer
from config import LEARNING_RATE, EPOCHS, BATCH_SIZE


def mean_squared_error_loss(weights, X_batch, y_batch):  # loss
    """Computes MSE loss over a batch of inputs."""
    predictions = linear_model_batch(weights, X_batch)  # fwd pass for the whole batch at once
    return np.mean((predictions - y_batch) ** 2)


def mse_gradient(weights, X_batch, y_batch):
    """Analytic gradient of the MSE loss w.r.t. weights (w) and bias (b)."""
    n = len(X_batch)
    w, b = weights[:-1], weights[-1]
    predictions = X_batch @ w + b
    error = predictions - y_batch  # d(loss)/d(prediction)

    grad_w = (2.0 / n) * (X_batch.T @ error)
    grad_b = (2.0 / n) * np.sum(error)

    return np.concatenate([grad_w, [grad_b]])


def calculate_accuracy(weights, X, y):
    """Computes binary classification accuracy (via sign of the linear output)."""
    preds = linear_model_batch(weights, X)  # run the whole dataset
    binary_preds = np.sign(preds)  # classify our output into + and - classes
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
            # -> after each epoch evaluate mse of improved model
            weights, loss = opt.step_and_cost(
                lambda w: mean_squared_error_loss(w, X_batch, y_batch),
                weights,
                lambda w: mse_gradient(w, X_batch, y_batch)
            )

        # Record loss over full dataset for the epoch
        epoch_loss = mean_squared_error_loss(weights, X, y)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = calculate_accuracy(weights, X, y)
            print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {acc * 100:.1f}%")

    return weights, loss_history
