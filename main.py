from datasets import load_academic_performance_data
from vqc import init_weights, draw_circuit
from training import train, calculate_accuracy
from visualizations import plot_training_loss, plot_decision_boundary
from config import PLOT_PATH, FEATURE_DIM

def main():
    print("=" * 60)
    print(" VQC with Intermediate Encoding (Academic Performance Data) ")
    print("=" * 60)
    
    # load the dataset  
    X, y = load_academic_performance_data(n_samples=100)
    print(f"Dataset loaded: {len(X)} student samples with {FEATURE_DIM} features.")
    
    # initialize weights
    weights = init_weights()
    print("\nCircuit Architecture:")
    print(draw_circuit(weights, X[0]))
    
    # train model
    print("\nStarting Quantum Optimization Pipeline...")
    trained_weights, loss_history = train(weights, X, y)
    
    # evaluation
    final_acc = calculate_accuracy(trained_weights, X, y)
    print(f"\nFinal Academic Classification Accuracy: {final_acc * 100:.2f}%")
    
    # visualizations
    plot_training_loss(loss_history, save_path=PLOT_PATH)
    plot_decision_boundary(trained_weights, X, y)

if __name__ == "__main__":
    main()