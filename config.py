#hyperparams independent of implementation
NUM_QUBITS = 4
NUM_LAYERS = 12
FEATURE_DIM = 4

LEARNING_RATE = 0.05
EPOCHS = 50 #complete pass through the entire training dataset, Multiple epochs allow the optimizer to progressively improve the variational parameters
BATCH_SIZE = 8 #instead of waiting until we've processed the entire dataset, batches allow the optimizer to update the model more frequently.
SEED = 42 #VQC starts with randomly initialized weights (rotation angles)

ENCODING_METHOD = "rx_rz"

PLOT_PATH = "training_loss.png"
