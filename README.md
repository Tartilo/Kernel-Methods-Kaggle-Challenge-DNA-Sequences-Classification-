# Kernel-Methods-Kaggle-Challenge-DNA-Sequences-Classification-

## 1. Kaggle Challenge Description
This project is part of the Kaggle challenge **"Data Challenge Kernel Methods 2024-2025."** The goal is to classify DNA sequences using kernel-based machine learning methods. We explore different kernel functions, including the **Spectrum Kernel**, to compute similarity between sequences and use **Kernelized Logistic Regression** for classification.

https://www.kaggle.com/competitions/data-challenge-kernel-methods-2024-2025/data

---

## 2. Structure of the Repository


| **Folder/File**             | **Description**                                                                                                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `data/`                  | Dataset folder                                                                                                        |
| `README.md`              | Document explaining the project and its main functionalities.                                                                                                   |
| `Requirements.txt`               | List of dependencies                                                                                                                  |
| `compute_kernel_matrix.py`               | Kernel matrix computation                                                                                                                               |
| `kernels.py`            | Kernel functions implementation                                                                                                                  |
| `km_kaggle.ipynb`                 | Notebook for experimentation                                                                                                   |
| `logistic_classifier.py`                  | Kernel logistic regression implementation                                                                                                                     |
| `start.py`               | Main script to execute the pipeline                                                                                                                          |

---

## 2. Running the code
To execute the code, put the desired Kernels and parameters like the following example:
```bash
python start.py \
--kernels mismatch spectrum mismatch \
--k_values 10 8 8 \
--m_values 1 "None" 1 \
--lambda_decay_values "None" "None" "None" \
--reg_lambda 0.00001 0.00001 0.0001
