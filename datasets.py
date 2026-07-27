import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import SEED, NUM_QUBITS

def load_academic_performance_data(n_samples=200):
    """
    Loads student academic learning data (e.g., study time, 
    past failures, attendance, and coursework scores) for binary 
    pass (+1) vs fail (-1) classification.
    """
    np.random.seed(SEED)
    
    try:
        # load UCI Student Performance dataset from OpenML
        dataset = fetch_openml(name="Student-Performance-Dataset", version=1, as_frame=True, parser="auto")
        df = dataset.frame
        
        # select numeric academic features (study time, failures, absences, G1 score)
        feature_cols = ['studytime', 'failures', 'absences', 'G1']
        X_raw = df[feature_cols].values
        
        #binary target: pass (final grade G3 >= 10 -> +1) vs fail (G3 < 10 -> -1)
        y_raw = np.where(df['G3'].values >= 10, 1.0, -1.0)
        
        # sample down if needed for simulator efficiency
        indices = np.random.choice(len(X_raw), size=min(n_samples, len(X_raw)), replace=False)
        X_raw, y = X_raw[indices], y_raw[indices]

    except Exception:
        # fallback synthetic academic performance dataset if offline
        print("Loading offline synthetic academic performance dataset...")
        
        # features: [studytime (hours), past_failures, absences, reading_score]
        studytime = np.random.uniform(1.0, 10.0, n_samples)
        failures = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.7, 0.15, 0.1, 0.05])
        absences = np.random.exponential(scale=4, size=n_samples)
        reading_score = np.random.uniform(40, 100, n_samples)
        
        X_raw = np.column_stack([studytime, failures, absences, reading_score])
        
        #rule: passing correlated with higher study time & reading score, lower failures
        academic_index = (studytime * 1.5) + (reading_score * 0.1) - (failures * 3.0) - (absences * 0.2)
        y = np.where(academic_index > np.median(academic_index), 1.0, -1.0)

    # preprocessing for Quantum Circuit
    # a. Dimensional reduction to match circuit wires (NUM_QUBITS)
    if X_raw.shape[1] > NUM_QUBITS:
        pca = PCA(n_components=NUM_QUBITS)
        X_proj = pca.fit_transform(X_raw)
    else:
        X_proj = X_raw[:, :NUM_QUBITS]

    # b. standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proj)

    # c. scale angles to [-pi, pi] for quantum rotation gates (RX/RZ)
    X_quantum = np.clip(X_scaled, -np.pi, np.pi)

    return X_quantum, y

if __name__ == "__main__":
    X, y = load_academic_performance_data(10)
    print("Sample Academic Features (Quantum encoded angles):\n", X[:3])
    print("Sample Targets (1.0 = Pass, -1.0 = At Risk/Fail):\n", y[:3])