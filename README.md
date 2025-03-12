# Kernel-Methods-DNA-Sequences-Classification-

## 1. Kaggle Challenge Description
This project is part of the Kaggle challenge **"Data Challenge Kernel Methods 2024-2025."** The goal of this challenge is to predict whether a DNA sequence region is binding site to a specific transcription factor. 
All information about the Challenge can be found on the report and the following website:

https://www.kaggle.com/competitions/data-challenge-kernel-methods-2024-2025/data

---

## 2. Structure of the Repository


| **Folder/File**             | **Description**                                                                                   |
|-----------------------------|---------------------------------------------------------------------------------------------------|
| `data/`                     | Folder containing the dataset.                                                                   |
| `README.md`                 | Documentation explaining the project and its main functionalities.                              |
| `requirements.txt`           | List of dependencies.                                                                         |
| `compute_kernel_matrix.py`   | Script for computing the kernel matrix.                                                       |
| `kernels.py`                 | Implementation of kernel functions.                                                           |
| `km_kaggle.ipynb`           | Jupyter notebook for experimentation.                                                          |
| `logistic_classifier.py`     | Kernel logistic regression implementation.                                                    |
| `start.py`                   | Main script to execute the pipeline.                                                          |

---


## 3. Running the code

To execute the code, put the desired Kernels and parameters like the following example:
```bash
python start.py \
--kernels mismatch spectrum mismatch \
--k_values 10 8 8 \
--m_values 1 "None" 1 \
--lambda_decay_values "None" "None" "None" \
--reg_lambda 0.00001 0.00001 0.0001
```

### **Explanation of Parameters**
| **Argument**                 | **Description**                                                                                         | **Example Values**             |
|------------------------------|---------------------------------------------------------------------------------------------------------|--------------------------------|
| `--kernels`                  | Names of the Kernels                            | `spectrum`, `mismatch`, `mismatch` |
| `--k_values`                 | Substring length (`k`) for each kernel.                                      | `10 8 8`                      |
| `--m_values`                 | Mismatch value (`m`) for Mismatch kernel. Use `none` for non-mismatch kernels.                      | `1 none 1`                    |
| `--lambda_decay_values`      | Decay values for Substring Kernel. Use `none` if not used.                                             | `"None" "None" "None"`              |
| `--reg_lambda`               | Regularization parameter for kernel logistic regression.                      | `0.00001 0.00001 0.0001`      |


## 4. References

Kernel Methods Course taught at the MVA by Julien Mairal, Michael Arbel and Jean-Philippe Vert.

https://mva-kernel-methods.github.io/course-page/

