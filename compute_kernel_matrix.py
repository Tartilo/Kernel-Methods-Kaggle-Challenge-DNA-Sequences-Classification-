import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from kernels import spectrum_kernel, mismatch_kernel, substring_kernel


def normalize_matrix(K):
    ''' We normalize the matrix for numerical stability'''
    root_K=np.sqrt(np.diag(K))
    K_inv=np.diag(1/root_K)
    return K_inv @ K @ K_inv

def compute_kernel_matrix_parallel(X, k, m=None, lambda_decay = None, n_jobs=-1):
    """
    Computes the Kernel matrix in parallel using joblib.
    """
    n = len(X)
    K = np.zeros((n, n))

    # Generate index pairs for upper triangle computation
    indices = [(i, j) for i in range(n) for j in range(i, n)]

    if lambda_decay is not None:
        # Compute kernel entries in parallel with tqdm
        results = Parallel(n_jobs=n_jobs)(
            delayed(substring_kernel)(X[i], X[j], k, lambda_decay) for i, j in tqdm(indices, desc="Computing Substring Kernel Matrix", unit=" entry")
        )

    elif m is not None:
        # Compute kernel entries in parallel with tqdm
        results = Parallel(n_jobs=n_jobs)(
            delayed(mismatch_kernel)(X[i], X[j], k, m) for i, j in tqdm(indices, desc="Computing Mismatch Kernel Matrix", unit=" entry")
        )

    else:
        # Compute kernel entries in parallel with tqdm
        results = Parallel(n_jobs=n_jobs)(
            delayed(spectrum_kernel)(X[i], X[j], k) for i, j in tqdm(indices, desc="Computing Spectrum Kernel Matrix", unit=" entry")
        )
        
    # Fill the kernel matrix
    index = 0
    for i, j in indices:
        K[i, j] = results[index]
        index += 1

    # Mirror the upper triangle to the lower triangle
    K = K + K.T - np.diag(K.diagonal())  # Ensure symmetry

    return normalize_matrix(K)
