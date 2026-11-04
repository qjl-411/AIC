DMSA-Net coexist-friendly final experiment files

These files are designed to live in the SAME folder as your completed 5-fold scripts.

Add these files:
- final_train.py
- final_test.py
- pcv_test_manifest_SEALED.csv

Shared file:
- pcv_v2_folds_seed42.csv

If your 5-fold folder already contains pcv_v2_folds_seed42.csv, do NOT keep a second differently named copy.
The final scripts intentionally reuse that same frozen manifest.

Recommended single-folder layout:

data_prepare_check.py
smoke_test.py
train_5fold.py
pcv_v2_folds_seed42.csv
final_train.py
final_test.py
pcv_test_manifest_SEALED.csv

Run final stage:
1. final_train.py
2. final_test.py

Do not run final_test.py before final_train.py has produced final_model_epoch126.pth.
