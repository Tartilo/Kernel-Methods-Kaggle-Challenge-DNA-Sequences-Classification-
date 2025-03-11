import numpy as np
import pandas as pd
import argparse
from compute_kernel_matrix import compute_kernel_matrix_parallel
from logistic_classifier import train_kernel_logistic_regression, predict_kernel_logistic_binary

# Define fixed dataset paths
TRAIN_FILES = ["data/Xtr0.csv", "data/Xtr1.csv", "data/Xtr2.csv"]
TEST_FILES = ["data/Xte0.csv", "data/Xte1.csv", "data/Xte2.csv"]
LABELS_FILES = ["data/Ytr0.csv", "data/Ytr1.csv", "data/Ytr2.csv"]

def load_dataset(train_file, test_file, label_file):
    """Loads a dataset: training sequences, labels, and test sequences."""
    df_Xtr = pd.read_csv(train_file)
    df_Ytr = pd.read_csv(label_file)
    df_Xte = pd.read_csv(test_file)

    X_train = df_Xtr.iloc[:, 1].tolist()  # Extract training sequences
    y_train = df_Ytr.iloc[:, 1].values  # Extract labels
    X_test = df_Xte.iloc[:, 1].tolist()  # Extract test sequences

    # Convert labels from {0,1} to {-1,1}
    y_train = 2 * y_train - 1 if np.min(y_train) == 0 else y_train  

    return X_train, y_train, X_test

def main(args):
    """Runs kernel logistic regression for each dataset separately and concatenates predictions."""
    all_predictions = []
    args.m_values = [None if m == "None" else int(m) for m in args.m_values]
    args.lambda_decay_values = [None if ld == "None" else float(ld) for ld in args.lambda_decay_values]
    args.reg_lambda = [float(lmbd) for lmbd in args.reg_lambda]  # Ensure correct type

    for i in range(3):  # Loop over three datasets
        print(f"\nProcessing dataset {i}...")

        # Load dataset using fixed paths
        X_train, y_train, X_test = load_dataset(TRAIN_FILES[i], TEST_FILES[i], LABELS_FILES[i])
        kernel_type = args.kernels[i]

        # Select kernel-specific parameters
        if kernel_type == "mismatch":
            m = args.m_values[i]
            lambda_decay = None
        elif kernel_type == "substring":
            lambda_decay = args.lambda_decay_values[i]
            m = None
        else:  # spectrum
            m, lambda_decay = None, None

        # Compute kernel matrix
        print(f"Computing {kernel_type} kernel matrix for dataset {i}...")
        K_train = compute_kernel_matrix_parallel(X_train, args.k_values[i], m=m, lambda_decay=lambda_decay)

        # Train logistic regression model
        print(f"Training Kernel Logistic Regression model for dataset {i}...")
        alpha = train_kernel_logistic_regression(K_train, y_train, args.reg_lambda[i])


        # Make predictions
        print(f"Making predictions for dataset {i}...")
        y_pred = predict_kernel_logistic_binary(alpha, X_train, X_test, args.k_values[i], m=m, lambda_decay=lambda_decay)

        # Convert predictions to {0,1} format
        y_pred = ((y_pred + 1)/2).astype(int)
        all_predictions.append(y_pred)

    # Concatenate all predictions
    final_predictions = np.concatenate(all_predictions)

    # Save predictions to CSV file
    print("Saving final predictions to Yte.csv...")
    df_Yte = pd.DataFrame(data=final_predictions, columns=['Bound'])
    df_Yte.index.name = 'Id'
    df_Yte.to_csv('Yte.csv')

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel Logistic Regression Pipeline for Multiple Datasets")

    parser.add_argument("--kernels", nargs=3, type=str, choices=["spectrum", "substring", "mismatch"], required=True, help="Kernel type for each dataset (3 values)")
    parser.add_argument("--k_values", nargs=3, type=int, required=True, help="k-mer length for each dataset (3 values)")
    parser.add_argument("--m_values", nargs=3, type=str, default=["None", "None", "None"], help="Number of mismatches for each dataset (use 'None' for non-mismatch kernels)")
    parser.add_argument("--lambda_decay_values", nargs=3, type=str, default=["None", "None", "None"], help="Decay factor for each dataset (use 'None' for non-substring kernels)")
    parser.add_argument("--reg_lambda", nargs=3, type=float, default=[1e-5, 1e-5, 1e-5], help="Regularization parameter for logistic regression (3 values, default: 1e-5)")

    args = parser.parse_args()
    main(args)
