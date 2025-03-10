import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import tqdm
from kernels import spectrum_kernel, substring_kernel, mismatch_kernel


def kernel_logistic_loss(alpha, K, y, reg_lambda):
    """
    Computes the objective function for kernel logistic regression.
    """
    n = len(y)
    K_alpha = K @ alpha  
    
    loss = np.sum(np.log(1 + np.exp(-y * K_alpha))) / n  
    reg = (reg_lambda / 2) * (alpha @ K @ alpha)  
    return loss + reg

def kernel_logistic_gradient(alpha, K, y, reg_lambda):
    """
    Computes the gradient of the kernel logistic loss.
    """
    n = len(y)
    K_alpha = K @ alpha
    probs = 1 / (1 + np.exp(y * K_alpha))  

    grad = - (K @ (y * probs)) / n  
    grad += reg_lambda * (K @ alpha)  
    return grad


def train_kernel_logistic_regression(K, y, reg_lambda):
    """
    Solves for α using gradient-based optimization.
    """
    n = len(y)
    alpha0 = np.zeros(n)  # Initialize α to zeros

    # Use L-BFGS optimizer
    result = minimize(kernel_logistic_loss, alpha0, 
                      args=(K, y, reg_lambda), 
                      jac=kernel_logistic_gradient,
                      method='L-BFGS-B',
                      options={'maxiter': 1000})

    return result.x  # Optimized α values


def predict_kernel_logistic_binary(alpha, X_train, X_test, k, m = None, lambda_decay=None, n_jobs=-1):
    """
    Predicts binary labels using kernel logistic regression with parallel processing (for faster computation).
    """
    n_test, n_train = len(X_test), len(X_train)
    K_test = np.zeros((n_test, n_train))

    indices = [(i, j) for i in range(n_test) for j in range(n_train)]

    if lambda_decay is not None:
        results = Parallel(n_jobs=n_jobs)(
            delayed(substring_kernel)(X_test[i], X_train[j], k, lambda_decay) 
            for i, j in tqdm(indices, desc="Computing Test-Train Substring Kernel", unit=" entry")
        )

    elif m is not None:
            results = Parallel(n_jobs=n_jobs)(
                delayed(mismatch_kernel)(X_test[i], X_train[j], k, m) 
                for i, j in tqdm(indices, desc="Computing Test-Train Mismatch Kernel", unit=" entry")
            )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(spectrum_kernel)(X_test[i], X_train[j], k) 
            for i, j in tqdm(indices, desc="Computing Test-Train Spectrum Kernel", unit=" entry")
        )

    for index, (i, j) in enumerate(indices):
        K_test[i, j] = results[index]

    scores = K_test @ alpha  

    return np.sign(scores)  # Convert to {-1,1} labels
