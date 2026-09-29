import numpy as np
from config import FEATURE_DIM


def sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )


def logistic_model(weights, x):
    """
    Logistic regression model: p = sigmoid(w . x + b)

    This replaces the variational quantum circuit (vqc_circuit) from the
    original project. Where the VQC repeatedly re-encoded the classical
    features into rotation gates across multiple entangling layers, here
    the "encoding" collapses to a single linear combination of the raw
    features, squashed through a sigmoid into a class-1 probability --
    no quantum state, no layers, no entanglement.
    """
    w, b = weights[:-1], weights[-1]
    z = np.dot(w, x) + b
    return sigmoid(z)


def logistic_model_batch(weights, X):
    """Vectorized forward pass over a batch of samples: shape (n, features)."""
    w, b = weights[:-1], weights[-1]
    z = X @ w + b
    return sigmoid(z)


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
        f"p = sigmoid({terms} + ({b:.3f}))\n"
        f"Example input : {sample_x}\n"
        f"Example output: {logistic_model(weights, sample_x):.4f} (probability of Pass)"
    )

Claude finished the response
1785624691939_image.png
plot a line graphs using only lines and dark blue color for the graph using x-axis for encoding layers and the y axis for training and test accuracy percentages for the following data and the 2 curves will represent these values


Ran 2 commands, created a file, read a file
Ran 2 commands, created a file, read a file
Here's the line graph — training accuracy as a solid line, testing accuracy as a dashed line, both in dark blue, with layer count on the x-axis.


Accuracy vs layers
Image · PNG 
Investigating the Effect of Repeated Intermediate data Encoding.docx.pdf
pdf

rate this paper for me


Recalled memory
Recalled memory
Solid piece of work overall — here's an honest read, strengths and weaknesses:

Overall: 7.5/10 — well above what I'd expect from a routine internship assignment, with one significant methodological gap that limits how much weight the conclusions can bear.

What's strong

Framing and novelty of the question. Isolating "does data re-uploading still help when N=n" from the usual "re-uploading solves qubit scarcity" narrative is a genuinely interesting angle, and you motivate it clearly against the literature (the Alamo et al. re-uploading paper) rather than just restating it.
Controlled design. Holding everything fixed except depth, and stating that explicitly, is exactly the right instinct for isolating a single causal factor.
Mathematical formulation. The layered-unitary formalism (U(x) once vs. U_l(x) per layer) is clean and correctly distinguishes the two encoding regimes.
Honest discussion. You don't oversell — the train/test divergence is interpreted correctly as capacity vs. generalization, tied to standard statistical learning theory, and you resist the temptation to claim monotonic improvement where the data doesn't show it.
Structure. Full IMRaD-plus-hypotheses format, explicit RQs and H1–H3, a limitations section that pre-empts obvious reviewer objections — this reads like a real paper, not a lab report.
What holds it back

Statistical power is the big one. Your test set is 20 samples. A single misclassification is a 5-point swing in "accuracy." The differences you're interpreting as a meaningful trend — 70% → 45% → 60% → 55% — could easily be sampling noise on 20 points rather than a real generalization curve. You flag this in "Limitations" as future work (multiple seeds), but given how central the train/test divergence is to your entire narrative, this isn't a minor caveat — it's the thing that determines whether H2/H3 are actually supported. I'd want at least a few random seeds and error bars before trusting the shape of that curve, not just as future work.
No classical baseline. Since your own framing is "does this quantum-specific mechanism actually add value," a same-size logistic regression or MLP baseline (which I know is already on your list) would let you say something about whether the model is learning anything qubit/encoding-specific at all, versus just fitting 4 features on 200 samples.
Single train/test split. Related to the above — no cross-validation, so you can't separate "this depth generalizes worse" from "this particular 80/20 split happened to be harder."
Some experimental details are underspecified for reproducibility: actual learning rate, batch size, number of epochs are all described as "fixed" but never given a value in the text (only that they're constant across runs).
Duplicate "6. Experimental Design" section — looks like a leftover from drafting/formatting (the content isn't fully identical between the two but overlaps heavily); worth cleaning up before this goes anywhere formal.
Bottom line: the writing, framing, and formalism are genuinely paper-quality. The one thing I'd fix before calling it done is exactly what you've already scoped as future work — multiple seeds with mean ± std — because right now the central empirical claim (generalization peaks early, declines with depth) rests on single-run numbers from a 20-sample test set, and that's a thin foundation for the strongest claims in the abstract and conclusion.

logistic_regressionbaseline2.png
logistic_regressionbaseline.png
here are the two logistic regression graphs i got, can i use these as a baseline? but there are no reuploading or intermediate layers here right?


Scrutinized baseline validity and flagged visualization concerns requiring clarification
Scrutinized baseline validity and flagged visualization concerns requiring clarification
Yes, you're reading that correctly — logistic regression has no notion of "depth" or repeated encoding at all. It's a single flat linear decision boundary computed once, not a stack of layers, so there's nothing analogous to L=1,2,3...12 for it. That's actually fine and expected: the LR baseline isn't supposed to have a depth axis. It's meant to be one horizontal reference line you can draw across your Table 1 chart — "here's what a simple linear model gets, does any VQC depth actually beat this?" — rather than another depth-sweep curve.

So yes, you can use this as a baseline, with a couple of things to check first:

1. Feature count mismatch — this is the important one. Your VQC paper uses 4 features (studytime, failures, absences, G1) mapped to 4 qubits. The decision boundary plot only shows "Feature 1" vs "Feature 2" — two axes. Is the logistic regression actually trained on all 4 features (and this plot is just a 2D slice/projection for visualization, with the other two held fixed)? Or was it trained on only 2 features? If it's the latter, this isn't a fair apples-to-apples baseline against your 4-feature VQC — you'd want to retrain LR on the same 4 standardized features you used for the quantum circuit.

2. You're missing the number that actually matters. The loss curve shows healthy convergence (0.70 → ~0.22 BCE over 50 epochs, looks clean, no signs of instability), but for it to work as a baseline in your paper you need the same metric as Table 1: training accuracy % and testing accuracy % on the same 80/20 split. Right now I can't tell if LR beats your VQC or not — that single pair of numbers is what goes in a new row (or a horizontal dashed line) on your accuracy-vs-depth figure.

Also worth a quick sanity check: the decision boundary plot shows a pretty small number of distinct points — is that the full 200-sample dataset, or a small subset used just for that visualization? If it's the latter, that's totally fine for a plot, just don't let it get confused with the actual train/test split used to compute accuracy.

If you can pull the accuracy numbers, I can help you add LR as a reference line on the existing depth chart.

here's the visualizations.py

import matplotlib.pyplot as plt
import numpy as np
from logistic_model import logistic_model_batch


def plot_training_loss(loss_history, save_path=None):
    """Plots optimization loss curve."""
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, marker='o', color='purple', linewidth=2)
    plt.title("Logistic Regression Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
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

    probs = logistic_model_batch(weights, grid)
    Z = probs.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.8, vmin=0, vmax=1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="P(Pass)")
    plt.show()


def compute_accuracy(weights, X, y):
    probs = logistic_model_batch(weights, X)
    preds = (probs >= 0.5).astype(int)
    # adjust if your labels are {-1, +1} rather than {0, 1}
    return (preds == y).mean() * 100


