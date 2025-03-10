import numpy as np
import pandas as pd
import argparse
from compute_kernel_matrix import compute_kernel_matrix_parallel
from logistic_classifier import train_kernel_logistic_regression, predict_kernel_logistic_binary

def load_dataset(train_file, test_file, label_file):
    """Loads a single dataset: training sequences, labels, and test sequences."""
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
    """Main function to execute kernel logistic regression for each dataset separately and concatenate predictions."""
    
    all_predictions = []
    args.m_values = [None if m == "None" else int(m) for m in args.m_values]
    args.lambda_decay_values = [None if ld == "None" else float(ld) for ld in args.lambda_decay_values]

    for i in range(3):  # Loop over three datasets
        print(f"\nProcessing dataset {i}...")

        train_file = args.train_files[i]
        test_file = args.test_files[i]
        label_file = args.labels_files[i]
        kernel_type = args.kernels[i]  # Get the kernel for this dataset

        # Load dataset
        X_train, y_train, X_test = load_dataset(train_file, test_file, label_file)

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
        K_train = compute_kernel_matrix_parallel(X_train, args.k_values[i], m=m, lambda_decay=lambda_decay, n_jobs=-1)

        # Train logistic regression model
        print(f"Training Kernel Logistic Regression model for dataset {i}...")
        alpha = train_kernel_logistic_regression(K_train, y_train, args.reg_lambda)

        # Compute test-train kernel matrix
        print(f"Computing test-train kernel matrix for dataset {i}...")
        K_test = compute_kernel_matrix_parallel(X_test, args.k_values[i], m=m, lambda_decay=lambda_decay, n_jobs=-1)

        # Make predictions
        print(f"Making predictions for dataset {i}...")
        y_pred = predict_kernel_logistic_binary(alpha, X_train, X_test, args.k_values[i], m=m, lambda_decay=lambda_decay, n_jobs=-1)

        # Convert predictions to {0,1} format
        y_pred = (y_pred + 1) // 2
        all_predictions.append(y_pred)

    # Concatenate all predictions
    final_predictions = np.concatenate(all_predictions)

    # Save predictions to CSV file
    print("Saving final predictions to Yte.csv...")
    df_Yte = pd.DataFrame(data=final_predictions, columns=['Bound'])
    df_Yte.index.name = 'Id'
    df_Yte.to_csv('Yte.csv', index=True)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel Logistic Regression Pipeline for Multiple Datasets with Different Kernels")
    
    parser.add_argument("--train_files", nargs=3, type=str, required=True, help="Paths to three training data CSV files")
    parser.add_argument("--test_files", nargs=3, type=str, required=True, help="Paths to three test data CSV files")
    parser.add_argument("--labels_files", nargs=3, type=str, required=True, help="Paths to three training labels CSV files")
    
    parser.add_argument("--kernels", nargs=3, type=str, choices=["spectrum", "substring", "mismatch"], required=True, help="Kernel type for each dataset (3 values)")
    parser.add_argument("--k_values", nargs=3, type=int, required=True, help="k-mer length for each dataset (3 values)")
    
    parser.add_argument("--m_values", nargs=3, type=str, default=["None", "None", "None"], help="Number of mismatches for each dataset (use 'None' for non-mismatch kernels)")
    parser.add_argument("--lambda_decay_values", nargs=3, type=str, default=["None", "None", "None"], help="Decay factor for each dataset (use 'None' for non-substring kernels)")

    parser.add_argument("--reg_lambda", type=float, default=1e-4, help="Regularization parameter for logistic regression")

    args = parser.parse_args()
    main(args)
