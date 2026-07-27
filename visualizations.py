import matplotlib.pyplot as plt
import numpy as np
from vqc import vqc_circuit

def plot_training_loss(loss_history, save_path=None):
    """Plots optimization loss curve."""
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, marker='o', color='purple', linewidth=2)
    plt.title("VQC Training Loss (Intermediate Encoding)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

def plot_decision_boundary(weights, X, y):
    """Visualizes model decision regions across a 2D grid."""
    x_min, x_max = -1.2, 1.2
    y_min, y_max = -1.2, 1.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = np.array([vqc_circuit(weights, pt) for pt in grid])
    Z = preds.reshape(xx.shape)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    plt.title("Quantum Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="QNode Output")
    plt.show()