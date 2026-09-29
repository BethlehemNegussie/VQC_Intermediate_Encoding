from datasets import load_academic_performance_data
from logistic_model import init_weights, describe_model
from training import train, calculate_accuracy
from visualizations import plot_training_loss, plot_decision_boundary
from config import PLOT_PATH, FEATURE_DIM


def main():

    print("=" * 60)
    print(" Logistic Regression Model (Academic Performance Data) ")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_academic_performance_data(
        n_samples=100
    )

    print(
        f"Training samples: {len(X_train)} | "
        f"Testing samples: {len(X_test)}"
    )

    print(
        f"Features: {FEATURE_DIM}"
    )

    weights = init_weights()

    print("\nModel:")

    print(
        describe_model(
            weights,
            X_train[0]
        )
    )

    print("\nStarting Optimization...")

    trained_weights, loss_history = train(
        weights,
        X_train,
        y_train
    )

    train_acc = calculate_accuracy(
        trained_weights,
        X_train,
        y_train
    )

    test_acc = calculate_accuracy(
        trained_weights,
        X_test,
        y_test
    )

    print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
    print(f"Testing Accuracy : {test_acc * 100:.2f}%")

    plot_training_loss(
        loss_history,
        save_path=PLOT_PATH
    )

    plot_decision_boundary(
        trained_weights,
        X_train,
        y_train
    )


if __name__ == "__main__":
    main()
