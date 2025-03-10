# Kernel-Methods-Kaggle-Challenge-DNA-Sequences-Classification-

python start.py \
--train_files data/Xtr0.csv data/Xtr1.csv data/Xtr2.csv \
--test_files data/Xte0.csv data/Xte1.csv data/Xte2.csv \
--labels_files data/Ytr0.csv data/Ytr1.csv data/Ytr2.csv \
--kernels mismatch spectrum mismatch \
--k_values 8 8 8 \
--m_values 1 "None" 1 \
--lambda_decay_values "None" "None" "None"




