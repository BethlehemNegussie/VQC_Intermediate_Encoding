import numpy as np
from config import FEATURE_DIM


def linear_model(weights, x):
    """
    Simple linear regression model: y = w . x + b

    This replaces the variational quantum circuit (vqc_circuit) from the
    original project. Where the VQC repeatedly re-encoded the classical
    features into rotation gates across multiple entangling layers, here
    the "encoding" collapses to a single linear combination of the raw
    features plus a bias term -- there is no quantum state, no layers,
    and no entanglement, just a dot product.
    """
    w, b = weights[:-1], weights[-1]
    return np.dot(w, x) + b


def linear_model_batch(weights, X):
    """Vectorized forward pass over a batch of samples: shape (n, features)."""
    w, b = weights[:-1], weights[-1]
    return X @ w + b


def init_weights():
    np.random.seed(42)  # fix randomness for reproducibility
    # FEATURE_DIM weights + 1 bias term
    return np.random.uniform(
        -1,
        1,
        size=(FEATURE_DIM + 1,)
    )


def describe_model(weights, sample_x):
    """
    Text summary of the model, analogous to draw_circuit() in the
    original VQC project (which printed the quantum circuit diagram).
    """
    w, b = weights[:-1], weights[-1]
    terms = " + ".join(f"({wi:.3f} * x{i})" for i, wi in enumerate(w))
    return (
        f"y = {terms} + ({b:.3f})\n"
        f"Example input : {sample_x}\n"
        f"Example output: {linear_model(weights, sample_x):.4f}"
    )
