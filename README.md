# Kernel-Methods-Kaggle-Challenge-DNA-Sequences-Classification-


python start.py --train_files Xtr0.csv Xtr1.csv Xtr2.csv \
                --test_files Xte0.csv Xte1.csv Xte2.csv \
                --labels_files Ytr0.csv Ytr1.csv Ytr2.csv \
                --kernels mismatch spectrum mismatch \
                --k_values 6 5 6 \
                --m_values 1 None 1 \
                --lambda_decay_values None None None

