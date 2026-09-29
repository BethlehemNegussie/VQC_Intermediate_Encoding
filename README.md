# Variational Quantum Classifier with Intermediate Data Re-uploading and Classical Baselines

## Overview

This project investigates Variational Quantum Circuits (VQCs) with intermediate data re-uploading and compares their learning behavior with classical machine learning baselines.

The VQC repeatedly encodes classical features between trainable variational layers, allowing the input data to interact with the quantum circuit multiple times. The project also implements linear regression and logistic regression models as classical baselines.

The primary research focus is investigating how repeated intermediate encoding affects the expressive capacity and generalization performance of a VQC when the number of classical features matches the number of available qubits.

The classical models provide reference approaches for studying the behavior of the quantum model using conventional machine learning techniques.

## Project Components

This repository contains three models:

1. **Variational Quantum Classifier with Intermediate Encoding**
2. **Logistic Regression Baseline**
3. **Linear Regression Baseline**

Each model has its own implementation and training pipeline, while sharing the broader objective of investigating classification performance on the selected dataset.

---

## 1. Variational Quantum Classifier (VQC)

### Overview

The VQC is implemented using PennyLane and employs intermediate data re-uploading encoding.

Unlike conventional VQCs, which typically encode classical data once at the beginning of the circuit, this architecture repeatedly introduces the input features between trainable variational layers.

This enables the circuit to interact with the original classical information at multiple stages of its computation.

### Architecture

The quantum pipeline consists of the following steps:

1. Classical data
2. Feature scaling
3. Quantum data encoding
4. Variational quantum layer
5. Intermediate data re-uploading
6. Additional variational quantum layers
7. Measurement
8. Classical prediction

The circuit uses feature encoding, trainable quantum rotations, entangling CNOT gates, and measurement-based predictions.

### Motivation

Intermediate data re-uploading provides a way to increase the number of interactions between classical features and trainable quantum transformations without increasing the number of qubits.

This project investigates the effect of changing the number of repeated encoding stages while keeping the feature-to-qubit mapping fixed.

### Implemented Encoding Methods

The project README describes support for:

* Angle encoding
* Amplitude encoding
* Basis encoding
* Intermediate data re-uploading encoding

The experiments described in the accompanying research paper focus on RX-RZ intermediate encoding.

---

## 2. Logistic Regression Baseline

### Overview

The logistic regression model serves as a classical baseline for the VQC.

It replaces the quantum circuit with a classical linear model followed by a sigmoid activation function:

**p = sigmoid(w · x + b)**

where:

* `w` represents the model weights.
* `x` is the input feature vector.
* `b` is the bias.

The model is trained using binary cross-entropy loss and a hand-written Adam optimizer with analytic gradients and mini-batch gradient descent.

### Architecture

The logistic regression pipeline consists of the following steps:

1. Classical data
2. Feature scaling
3. Linear combination (`w · x + b`)
4. Sigmoid activation
5. Binary cross-entropy loss
6. Adam optimization
7. Binary classification

### Relationship to the VQC

The logistic regression implementation provides a classical reference model for binary classification.

Unlike the VQC, it does not use quantum gates, trainable quantum circuits, or repeated data encoding. Instead, it learns a linear decision function and converts its output into a probability using the sigmoid function.

The implementation aims to keep the broader data-processing and evaluation pipeline comparable to the VQC.

---

## 3. Linear Regression Baseline

### Overview

The linear regression model provides another classical reference for the VQC.

It replaces the quantum circuit with a linear function:

**y = w · x + b**

where:

* `w` represents the learned weights.
* `x` is the input feature vector.
* `b` is the bias.

The model is trained using mean squared error (MSE) loss and a hand-written Adam optimizer with analytic gradients and mini-batch gradient descent.

### Architecture

The linear regression pipeline consists of the following steps:

1. Classical data
2. Feature scaling
3. Linear combination (`w · x + b`)
4. Mean squared error loss
5. Adam optimization
6. Classical prediction

### Relationship to the VQC

The linear regression implementation provides a simple classical reference for examining the relationship between input features and predictions.

Unlike the VQC, it uses a single linear combination of the input features and does not have repeated data encoding or trainable quantum transformations.

Since the task is binary classification, the model's continuous outputs must be converted into class predictions using the project's chosen threshold before classification accuracy can be compared.

---

## Experimental Comparison

The three models provide different approaches to learning from classical data.

| Component             | VQC                                    | Logistic Regression          | Linear Regression            |
| --------------------- | -------------------------------------- | ---------------------------- | ---------------------------- |
| Model type            | Variational quantum circuit            | Classical linear classifier  | Classical linear model       |
| Main operations       | Quantum rotations and entangling gates | Weighted sum and sigmoid     | Weighted sum                 |
| Intermediate encoding | Yes                                    | No                           | No                           |
| Optimization          | Quantum-classical optimization         | Adam with analytic gradients | Adam with analytic gradients |
| Loss function         | MSE                                    | Binary cross-entropy         | MSE                          |
| Output                | Measurement-based prediction           | Probability                  | Continuous prediction        |
| Classification        | Sign-based prediction                  | Probability threshold        | Output threshold             |

For a meaningful experimental comparison, the models should use the same dataset, feature selection, preprocessing, and train/test split wherever possible. Differences in training objectives and model architectures should be documented when interpreting the results.

## Technology Stack

* Python
* PennyLane
* NumPy
* scikit-learn
* Matplotlib
* Pandas

## Project Structure

The repository contains separate implementations for the quantum model and the two classical baselines.

```text
VQC_Intermediate_Encoding/
├── vqc/
│   ├── main.py
│   ├── vqc.py
│   ├── feature_encoding.py
│   ├── ansatz.py
│   ├── training.py
│   ├── datasets.py
│   ├── visualization.py
│   └── config.py
│
├── baselines/
│   ├── logistic_regression/
│   │   ├── main.py
│   │   ├── logistic_model.py
│   │   ├── optim.py
│   │   ├── training.py
│   │   ├── datasets.py
│   │   ├── visualizations.py
│   │   └── config.py
│   │
│   └── linear_regression/
│       ├── main.py
│       ├── linear_model.py
│       ├── optim.py
│       ├── training.py
│       ├── datasets.py
│       ├── visualizations.py
│       └── config.py
│
├── paper/
│   └── intermediate_encoding_paper.pdf
│
├── README.md
└── requirements.txt
```

**Note:** The structure above is illustrative. Update it to match the actual repository layout before submitting the PR.

## Installation

Clone the repository:

```bash
git clone https://github.com/BethlehemNegussie/VQC_Intermediate_Encoding.git
cd VQC_Intermediate_Encoding
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Models

Run each model using its respective entry point. The exact commands depend on the final project structure.

## Research Paper

The accompanying research paper, *Investigating the Effect of Repeated Intermediate Data Encoding in Variational Quantum Circuits When the Number of Features Matches the Number of Available Qubits*, investigates the relationship between encoding depth, training performance, and testing performance.

The study uses four classical features mapped to four qubits and evaluates multiple circuit depths to investigate how repeated intermediate encoding affects model expressivity and generalization.

The classical baseline implementations provide additional reference models for the broader experimental investigation.

## Future Improvements

Potential extensions include:

* Comparing the VQC and classical baselines under identical data splits and preprocessing.
* Investigating different quantum encoding strategies.
* Evaluating additional variational circuit architectures.
* Testing on larger and geometrically structured datasets.
* Exploring regularization techniques for the classical baselines.
* Evaluating the VQC on real quantum hardware.
* Running multiple random seed initializations and reporting mean performance and standard deviation.
* Investigating gradient magnitudes at increasing circuit depths to study possible optimization limitations.

## License and Code Availability

The source code for the VQC and classical baseline implementations is maintained in this repository.

The accompanying research paper documents the experimental motivation, methodology, and results of the intermediate encoding depth study.
