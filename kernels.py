import numpy as np
from collections import Counter
from itertools import product

#
#Substring Kernel
#

def compute_B_matrix(x, x_prime, k, lambda_decay):
    """Computes the B_k matrix with dynamic programming"""
    n, m = len(x), len(x_prime)
    B = np.zeros((k + 1, n + 1, m + 1))
    B[0, :, :] = 1  # Base case

    for l in range(1, k + 1):
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if x[i - 1] == x_prime[j - 1]: 
                    B[l, i, j] = lambda_decay * (B[l, i - 1, j] + B[l, i, j - 1] - lambda_decay * B[l, i - 1, j - 1])
                    B[l, i, j] += (lambda_decay ** 2) * B[l - 1, i - 1, j - 1]
                else:
                    B[l, i, j] = lambda_decay * (B[l, i - 1, j] + B[l, i, j - 1] - lambda_decay * B[l, i - 1, j - 1])
    return B

def substring_kernel(x, x_prime, k, lambda_decay):
    """Computes the substring kernel K_k with dynamic programming"""
    n, m = len(x), len(x_prime)
    
    # Compute B_k
    B = compute_B_matrix(x, x_prime, k, lambda_decay)

    # Base cases
    if k==0:
        K = np.ones((n + 1, m + 1))
    else:
        K = np.zeros((n + 1, m + 1))

    # Dynamic Programming
    for i in range(1, n + 1):
        K[i, :] = K[i - 1, :]

        for j in range(1, m + 1):
            res = 0
            for j_prime in range(1,m+1):
                if x_prime[j_prime-1] == x[i - 1]:
                    res += B[k-1, i - 1, j_prime-1]
            K[i, j] += (lambda_decay**2) * res

    # Return the final kernel value
    return K[n, m]  

#
#Spectrum Kernel
#

def get_kmer_counts(seq, k):
    """Extracts k-mer counts from a DNA sequence."""
    return Counter([seq[i:i+k] for i in range(len(seq) - k + 1)])

def spectrum_kernel(x, x_prime, k):
    # Compute k-mer counts for both sequences
    kmer_counts_x = get_kmer_counts(x, k)
    kmer_counts_x_prime = get_kmer_counts(x_prime, k)

    # Compute dot product of k-mer count vectors
    return sum(kmer_counts_x[u] * kmer_counts_x_prime[u] for u in set(kmer_counts_x) & set(kmer_counts_x_prime))

#
#Mismatch Kernel
#

def hamming_distance(s1, s2):
    """Computes Hamming distance between two equal-length strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def generate_mismatch_kmers(kmer, m):
    """Generates all possible k-mers within `m` mismatches of a given k-mer."""
    bases = ['A', 'C', 'G', 'T']
    mismatch_kmers = set([kmer])

    for positions in product(range(len(kmer)), repeat=m):
        for replacements in product(bases, repeat=m):
            kmer_list = list(kmer)
            for pos, replacement in zip(positions, replacements):
                kmer_list[pos] = replacement
            mismatch_kmers.add("".join(kmer_list))

    return mismatch_kmers

def count_kmers_with_mismatches(sequence, k, m):
    """Counts k-mers in `sequence`, including up to `m` mismatches."""
    kmer_counts = {}

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        mismatch_kmers = generate_mismatch_kmers(kmer, m)

        for mismatch_kmer in mismatch_kmers:
            if mismatch_kmer in kmer_counts:
                kmer_counts[mismatch_kmer] += 1
            else:
                kmer_counts[mismatch_kmer] = 1

    return kmer_counts

def mismatch_kernel(x, x_prime, k, m):
    # Count k-mers with mismatches
    x_counts = count_kmers_with_mismatches(x, k, m)
    x_prime_counts = count_kmers_with_mismatches(x_prime, k, m)

    # Compute kernel similarity using dot product of k-mer counts
    similarity = sum(x_counts[kmer] * x_prime_counts.get(kmer, 0) for kmer in x_counts)

    return similarity
