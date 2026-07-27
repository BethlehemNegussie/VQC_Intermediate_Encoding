import pennylane as qml
from pennylane import numpy as np
from config import NUM_QUBITS, NUM_LAYERS, ENCODING_METHOD
from feature_encoding import FeatureEncoder, variational_layer

dev = qml.device("default.qubit", wires=NUM_QUBITS)

encoder = FeatureEncoder()

@qml.qnode(dev)
def vqc_circuit(weights, x):
    for layer in range(NUM_LAYERS):
        encoder.intermediate_encode(
            x,
            wires=range(NUM_QUBITS)
        )
        qml.Barrier()
        variational_layer(
            weights[layer],
            wires=range(NUM_QUBITS)
        )
        qml.Barrier()

    return qml.expval(qml.PauliZ(0))


def init_weights():
    np.random.seed(42)
    return np.random.uniform(
        0,
        2 * np.pi,
        size=(NUM_LAYERS, NUM_QUBITS, 3),
        requires_grad=True
    )


def draw_circuit(weights, sample_x):
    return qml.draw(vqc_circuit)(weights, sample_x)
