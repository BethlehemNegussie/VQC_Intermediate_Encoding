# Variational Quantum Classifier with Intermediate Data Re-uploading Encoding

## Overview

This project implements a Variational Quantum Classifier (VQC) using PennyLane with intermediate data re-uploading encoding.

Traditional VQCs usually encode classical data once at the beginning of the circuit and then apply variational transformations. This implementation explores an alternative architecture where classical information is repeatedly injected between variational layers.

The goal is to investigate how intermediate data re-uploading can improve information flow between classical features and trainable quantum parameters while working under limited qubit resources.

---

## Architecture

The implemented pipeline is:

Classical Data  
↓  
Data Encoding  
↓  
Quantum State Preparation  
↓  
Variational Layer  
↓  
Intermediate Data Re-uploading  
↓  
Variational Layer  
↓  
Measurement  
↓  
Classical Prediction


The circuit consists of:

- Feature encoding layers
- Trainable quantum rotation layers
- Entangling CNOT gates
- Measurement-based prediction

---

## Motivation

Near-term quantum devices are constrained by limited qubit availability and shallow circuit depth.

Encoding data only once can limit how frequently classical information interacts with trainable quantum parameters.

Intermediate data re-uploading addresses this by repeatedly introducing classical information throughout the circuit, allowing different variational layers to process the input multiple times.

---

## Implemented Encoding Methods

The project supports:

- Angle encoding
- Amplitude encoding
- Basis encoding
- Intermediate data re-uploading encoding

---

## Tech stack

- Python
- PennyLane
- NumPy
- Machine Learning optimization methods

---

## Project Structure
```bash
├── main.py
├── vqc.py
├── feature_encoding.py
├── ansatz.py
├── training.py
├── datasets.py
├── visualization.py
├── config.py
└── requirements.txt
```
---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd VQC-Intermediate-Data-Reuploading
```
---
Install dependencies
```bash
pip install -r requirements.txt
```
Run the project
```bash
python main.py
```

## Future Improvements
```bash
Possible extensions:

-Comparison of different encoding strategies
-Testing on larger datasets
-Experiments with different ansatz architectures
-Evaluation on real quantum hardware
```