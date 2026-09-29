import matplotlib.pyplot as plt
import numpy as np
from linear_model import linear_model_batch


def plot_training_loss(loss_history, save_path=None):
    """Plots optimization loss curve."""
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, marker='o', color='purple', linewidth=2)
    plt.title("Linear Regression Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_decision_boundary(weights, X, y):
    """Visualizes model decision regions across a 2D grid (first two features)."""
    x_min, x_max = -1.2, 1.2
    y_min, y_max = -1.2, 1.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))

    # Remaining features (beyond the first two) are held at their mean so the
    # 2D slice stays comparable to the original quantum decision-boundary plot.
    other_means = X[:, 2:].mean(axis=0) if X.shape[1] > 2 else np.array([])

    grid = np.c_[xx.ravel(), yy.ravel()]
    if other_means.size:
        extra = np.tile(other_means, (grid.shape[0], 1))
        grid = np.hstack([grid, extra])

    preds = linear_model_batch(weights, grid)
    Z = preds.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    plt.title("Linear Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Model Output")
    plt.show()
