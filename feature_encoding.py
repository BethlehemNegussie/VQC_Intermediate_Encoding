import pennylane as qml

class FeatureEncoder:
    def angle_encode(self, x, wires):
        for feature, wire in zip(x, wires):
            qml.RY(feature, wires=wire)

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

    def intermediate_encode(self, x, wires):
        self.rx_rz_encode(x, wires)


def variational_layer(weights_layer, wires):
    for i, wire in enumerate(wires):
        qml.RX(weights_layer[i, 0], wires=wire)
        qml.RY(weights_layer[i, 1], wires=wire)
        qml.RZ(weights_layer[i, 2], wires=wire)

    for i in range(len(wires)):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])
