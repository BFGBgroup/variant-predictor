# predict_variants.py
# Usage: python predict_variants.py input_variants.txt [output_dir]

import numpy as np
import pandas as pd
import os, sys, torch, random

# ─────────────────────────────────────────
# Automatic paths
# ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from integrao.integrater import integrao_predictor
except ImportError:
    print("Error: cannot import 'integrao.integrater'. Please ensure IntegrAO is installed in the environment.")
    sys.exit(1)

DATA_DIR = os.path.join(SCRIPT_DIR, "feature_db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATHS = [os.path.join(MODEL_DIR, f"model_fold{i}.pth") for i in range(1,6)]

# ─────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python predict_variants.py input_variants.txt [output_dir]")
    sys.exit(1)

input_file = sys.argv[1]
output_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.dirname(os.path.abspath(input_file))
os.makedirs(output_dir, exist_ok=True)

# ─────────────────────────────────────────
# Read input variants
# ─────────────────────────────────────────
with open(input_file) as f:
    query_ids = [line.strip() for line in f if line.strip()]
print(f"Input variants: {len(query_ids)}")

# ─────────────────────────────────────────
# Load feature database
# ─────────────────────────────────────────
enformer = pd.read_csv(os.path.join(DATA_DIR, "enformer_dbsnp_intersect_common.tsv"), sep="\t", index_col=0)
favor    = pd.read_csv(os.path.join(DATA_DIR, "favor_dbsnp_features_common.tsv"), sep="\t", index_col=0)
count_df = pd.read_csv(os.path.join(DATA_DIR, "dbsnp_features_count_common.tsv"), sep="\t", index_col=0)
dist_df  = pd.read_csv(os.path.join(DATA_DIR, "dbsnp_features_dist_common.tsv"), sep="\t", index_col=0)

# Take absolute values
enformer = enformer.abs()
favor    = favor.abs()
count_df = count_df.abs()
dist_df  = dist_df.abs()

# Drop duplicates
for name, df in [("enformer", enformer), ("favor", favor), ("count", count_df), ("dist", dist_df)]:
    if df.index.duplicated().any():
        print(f"  Warning: {name} contains duplicate rs_id, keeping the first occurrence")
    df.drop_duplicates(inplace=True)

# ─────────────────────────────────────────
# Filter input variants that exist in the database
# ─────────────────────────────────────────
available = set(query_ids) & set(enformer.index) & set(favor.index) & set(count_df.index) & set(dist_df.index)
missing = set(query_ids) - available
if missing:
    print(f"  Warning: {len(missing)} variants not found in the feature database, skipped:")
    for v in sorted(missing):
        print(f"    {v}")

valid_ids = [v for v in query_ids if v in available]
print(f"Valid variants: {len(valid_ids)}")
if len(valid_ids) == 0:
    print("No variants available for prediction. Exiting.")
    sys.exit(1)

X_enformer = enformer.loc[valid_ids]
X_favor    = favor.loc[valid_ids]
X_count    = count_df.loc[valid_ids]
X_dist     = dist_df.loc[valid_ids]

# ─────────────────────────────────────────
# Set random seed
# ─────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────
# Initialize predictor & network construction
# ─────────────────────────────────────────
predictor = integrao_predictor(
    [X_enformer, X_favor, X_count, X_dist],
    dataset_name="dbsnp_prediction",
    modalities_name_list=["enformer", "favor", "count", "dist"],
    neighbor_size=30,
    embedding_dims=10,
    fusing_iteration=20,
    normalization_factor=1.0,
    alighment_epochs=1000,
    beta=1.6,
    mu=0.6,
    num_classes=2
)
predictor.network_diffusion()

# ─────────────────────────────────────────
# 5-fold ensemble prediction
# ─────────────────────────────────────────
probs_all, preds_all = [], []

for model_path in MODEL_PATHS:
    preds, probs = predictor.inference_supervised(
        model_path,
        new_datasets=[X_enformer, X_favor, X_count, X_dist],
        modalities_names=["enformer", "favor", "count", "dist"]
    )[:2]
    probs_all.append(probs)
    preds_all.append(preds)

probs_mean  = np.mean(np.stack(probs_all, axis=0), axis=0)
preds_vote  = np.sum(np.stack(preds_all, axis=0), axis=0)
preds_final = (probs_mean >= 0.5).astype(int)

sample_ids = np.array(list(predictor.dict_sampleToIndexs.keys()))

# ─────────────────────────────────────────
# Save results
# ─────────────────────────────────────────
df_out = pd.DataFrame({
    "Variant":     sample_ids,
    "probs_mean":  probs_mean,
    "preds_final": preds_final,
    "preds_vote":  preds_vote,
})

out_path = os.path.join(output_dir, "prediction_results.tsv")
df_out.to_csv(out_path, sep="\t", index=False)
print(f"\nResults saved to: {out_path}")
print(f"Predicted 1: {preds_final.sum()}  Predicted 0: {(preds_final==0).sum()}")
