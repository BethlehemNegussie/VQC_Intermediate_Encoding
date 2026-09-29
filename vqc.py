import pennylane as qml
from pennylane import numpy as np #a wrapper that cretes arrays that support automatic differentiation
#compute gradients in a compatible way with classical techniques like backprop
from config import NUM_QUBITS, NUM_LAYERS, ENCODING_METHOD
from feature_encoding import FeatureEncoder, variational_layer

dev = qml.device("default.qubit", wires=NUM_QUBITS) # create quantum simulator 

encoder = FeatureEncoder() #encoder object creates an encoder from the encoder class

@qml.qnode(dev) #describe a quantum circuit that should be executed on this device
def vqc_circuit(weights, x): # weights = trainable theta form our ansatz
    for layer in range(NUM_LAYERS): # NUM_LAYERS for how many times the var layer and the encoding happen before measurement
        encoder.intermediate_encode( 
            #For every layer inject the classical data again, a new opportunity to combine new quantum state+original classical information
            x,
            wires=range(NUM_QUBITS)
        )
        qml.Barrier() #visual
        variational_layer( #applies your trainable gates for each layer 
            weights[layer],
            wires=range(NUM_QUBITS)
        )
        qml.Barrier()
    return qml.expval(qml.PauliZ(0))
    """
    CNOT causes entanglement between the features encoded in each qubits 
    and entanglement allows the circuit to learn these feature interactions
    measurement value converts the quantum states to classical vals in [-1, 1]
    because the Z expectation value naturally maps to the range [−1,1] which fits binary classification
    measure if qubit value is whether closer to state 0 or 1, 0 =+1, 1 =-1
    entangling operations allow information from all qubits to influence the final state of the measured qubit
    """
def init_weights():
    np.random.seed(42) #fix randomness for reproducibility
    return np.random.uniform(
        0,
        2 * np.pi, # 0- 2 pi sice rotations are periodic
        size=(NUM_LAYERS, NUM_QUBITS, 3), #layer 0 qubit 0 rx, ry, rz, kayer 1 qubit 1 rx, ry, rz
        requires_grad=True #tells pennylane to calculate the gradients
    )


def draw_circuit(weights, sample_x):
    return qml.draw(vqc_circuit)(weights, sample_x)
