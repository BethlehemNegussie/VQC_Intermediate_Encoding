import pennylane as qml

class FeatureEncoder:
    def angle_encode(self, x, wires): #single axis encoding 
        for feature, wire in zip(x, wires):
            qml.RY(feature, wires=wire)
    # encoding using two axes
    def rx_rz_encode(self, x, wires):
        for i, wire in enumerate(wires):
            feature = x[i % len(x)]
            qml.RX(feature, wires=wire)
            qml.RZ(feature, wires=wire)

    def amplitude_encode(self, x, wires):
        qml.AmplitudeEmbedding(
            x,
            wires=wires,
            normalize=True
        )

    def basis_encode(self, x, wires):
        qml.BasisEmbedding(
            x,
            wires=wires
        )

    def encode(self, x, wires, method="rx_rz"):
        if method == "angle":
            self.angle_encode(x, wires)
        elif method == "amplitude":
            self.amplitude_encode(x, wires)
        elif method == "basis":
            self.basis_encode(x, wires)
        elif method == "rx_rz":
            self.rx_rz_encode(x, wires)
        else:
            raise ValueError("Unsupported encoding method")
    #repeated encoding allows multiple trainable layers to interact directly with the classical information
    def intermediate_encode(self, x, wires):
        self.rx_rz_encode(x, wires)

#the rotation angles are trainable parameters optimized to minimize the loss function
#applies a fixed sequence of trainable rotation gates using the current variational parameters
def variational_layer(weights_layer, wires):
    for i, wire in enumerate(wires): #loop through each qubit, optimize and
        qml.RX(weights_layer[i, 0], wires=wire)
        qml.RY(weights_layer[i, 1], wires=wire)
        qml.RZ(weights_layer[i, 2], wires=wire)
#enatangle neighboring qubits so the model can learn relationships between features.
    for i in range(len(wires)):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]]) #Modulo is wrapping the loop back to the beginning layer creating a closed loop
