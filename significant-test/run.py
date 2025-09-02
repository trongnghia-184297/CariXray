import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import ttest_rel
import argparse
from src import *

parser = argparse.ArgumentParser()

parser.add_argument('--csv_path', dest='csv_path',
                    type=str, default="", help='')

parser.add_argument('--out_p', dest='out_p',
                    type=str, default="", help='')

parser.add_argument('--out_sig', dest='out_sig',
                    type=str, default="", help='')

args = parser.parse_args()

csv_path = args.csv_path
out_p = args.out_p
out_sig = args.out_sig

make_folder(os.path.dirname(out_p))

# -----------------------------
# Bước 1. Đọc dữ liệu (5 folds × 8 models)
# -----------------------------
# csv_path = "det.csv"  # change to your file if needed
# out_p = "det_pairwise_ttest_pvalues_raw.csv"
# out_sig = "det_pairwise_ttest_significant_mask.csv"
df = pd.read_csv(csv_path)

# Ensure correct ordering: first column is 'Fold', rest are model columns
model_cols = list(df.columns[1:])  # 8 model names
n_models = len(model_cols)
print("-"*50)
print("step 1 input data:")
print(df.head(10))
print("-"*50)



# -----------------------------
# Bước 2. Tạo ma trận chênh lệch cho từng fold (A - B)
# -----------------------------
diff_mats = {}  # fold index -> DataFrame (8×8)
for _, row in df.iterrows():
    fold_idx = int(row.iloc[0])
    values = row[model_cols].to_numpy(dtype=float)
    # build pairwise difference matrix: M[i,j] = values[i] - values[j]
    mat = values.reshape(-1,1) - values.reshape(1,-1)
    diff_mats[fold_idx] = pd.DataFrame(mat, index=model_cols, columns=model_cols)
print("step 2 diff matrix:")
print(diff_mats[1])
print("-"*50)

# -----------------------------
# Bước 3. Gom thành vector chênh lệch cho từng cặp (dài 5)
# -----------------------------
pair_vectors = {}  # (A,B) -> np.array of length 5 containing differences A-B per fold
for i, j in combinations(range(n_models), 2):
    A, B = model_cols[i], model_cols[j]
    d_vec = df[A].to_numpy(dtype=float) - df[B].to_numpy(dtype=float)  # length = 5 folds
    pair_vectors[(A, B)] = d_vec

# Build a small preview table of the first few pairs and their difference vectors
preview_rows = []
for k, ((A,B), vec) in enumerate(pair_vectors.items()):
    if k >= 6: break
    preview_rows.append({"Pair (A vs B)": f"{A} vs {B}", "d (5 folds)": list(np.round(vec, 3))})
preview_df = pd.DataFrame(preview_rows)
print("step 3 preview (first 6 pairs):")
print(preview_df)
print("-"*50)
# -----------------------------
# Bước 4. Paired t-test (two-sided, α=0.05) cho mọi cặp → Ma trận 8×8 p-value
# -----------------------------
p_mat = np.full((n_models, n_models), np.nan, dtype=float)
sig_mat = np.full((n_models, n_models), False, dtype=bool)

for i in range(n_models):
    for j in range(n_models):
        if i == j:
            continue
        # Two equivalent ways:
        # 1) paired t-test directly on the two model columns:
        #    t_stat, p_val = ttest_rel(df[model_cols[i]], df[model_cols[j]])
        # 2) one-sample t-test on the difference vector against 0 (ttest_rel internally does this).
        d = df[model_cols[i]].to_numpy(dtype=float) - df[model_cols[j]].to_numpy(dtype=float)
        # Using ttest_rel for clarity with paired design:
        t_stat, p_val = ttest_rel(df[model_cols[i]], df[model_cols[j]])
        p_mat[i, j] = p_val
        sig_mat[i, j] = (p_val < 0.05)

pvals_df = pd.DataFrame(p_mat, index=model_cols, columns=model_cols)
sig_df = pd.DataFrame(sig_mat, index=model_cols, columns=model_cols)

# Optional: a pretty-printed matrix where significant cells are marked with '*'
pretty = pvals_df.copy().astype(object)  # cho phép chứa string
for i in range(n_models):
    for j in range(n_models):
        if i == j:
            pretty.iloc[i, j] = "–"
        else:
            pv = pvals_df.iloc[i, j]
            pretty.iloc[i, j] = f"{pv:.4f}" + ("*" if pv < 0.05 else "")
pvals_rounded = pvals_df.round(4)


print("step 4 pvals:")
print(pvals_rounded)
print("-"*50)

pvals_rounded.to_csv(out_p)
sig_df.to_csv(out_sig)
