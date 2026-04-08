# predict_variants.py
# 用法: python predict_variants.py input_variants.txt [output_dir]

import numpy as np
import pandas as pd
import os, sys, torch, random

# ─────────────────────────────────────────
# 自动设置路径，无需用户配置
# ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# IntegrAO 安装路径假设在 site-packages，可直接导入
try:
    from integrao.integrater import integrao_predictor
except ImportError:
    print("Error: cannot import 'integrao.integrater'. Please ensure IntegrAO is installed in the environment.")
    sys.exit(1)

DATA_DIR = os.path.join(SCRIPT_DIR, "feature_db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATHS = [os.path.join(MODEL_DIR, f"model_fold{i}.pth") for i in range(1,6)]

# ─────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────
if len(sys.argv) < 2:
    print("用法: python predict_variants.py input_variants.txt [output_dir]")
    sys.exit(1)

input_file = sys.argv[1]
output_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.dirname(os.path.abspath(input_file))
os.makedirs(output_dir, exist_ok=True)

# ─────────────────────────────────────────
# 读取输入 variants
# ─────────────────────────────────────────
with open(input_file) as f:
    query_ids = [line.strip() for line in f if line.strip()]
print(f"输入 variants: {len(query_ids)}")

# ─────────────────────────────────────────
# 读取特征数据库并取绝对值
# ─────────────────────────────────────────
enformer = pd.read_csv(os.path.join(DATA_DIR, "enformer_dbsnp_intersect_common.tsv"), sep="\t", index_col=0)
favor    = pd.read_csv(os.path.join(DATA_DIR, "favor_dbsnp_features_common.tsv"), sep="\t", index_col=0)
count_df = pd.read_csv(os.path.join(DATA_DIR, "dbsnp_features_count_common.tsv"), sep="\t", index_col=0)
dist_df  = pd.read_csv(os.path.join(DATA_DIR, "dbsnp_features_dist_common.tsv"), sep="\t", index_col=0)

enformer = enformer.abs()
favor    = favor.abs()
count_df = count_df.abs()
dist_df  = dist_df.abs()

# 去重
for name, df in [("enformer", enformer), ("favor", favor), ("count", count_df), ("dist", dist_df)]:
    if df.index.duplicated().any():
        print(f"  警告: {name} 存在重复 rs_id，保留第一条")
    df.drop_duplicates(inplace=True)

# ─────────────────────────────────────────
# 取查询 variants 与数据库的交集
# ─────────────────────────────────────────
available = set(query_ids) & set(enformer.index) & set(favor.index) & set(count_df.index) & set(dist_df.index)
missing = set(query_ids) - available
if missing:
    print(f"  警告: {len(missing)} 个 variants 在特征库中找不到，已跳过:")
    for v in sorted(missing):
        print(f"    {v}")

valid_ids = [v for v in query_ids if v in available]
print(f"有效 variants: {len(valid_ids)}")
if len(valid_ids) == 0:
    print("没有可预测的 variants，退出。")
    sys.exit(1)

X_enformer = enformer.loc[valid_ids]
X_favor    = favor.loc[valid_ids]
X_count    = count_df.loc[valid_ids]
X_dist     = dist_df.loc[valid_ids]

# ─────────────────────────────────────────
# 随机种子
# ─────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────
# 初始化 predictor & 构图
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
# 5-fold ensemble
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
# 保存结果
# ─────────────────────────────────────────
df_out = pd.DataFrame({
    "Variant":     sample_ids,
    "probs_mean":  probs_mean,
    "preds_final": preds_final,
    "preds_vote":  preds_vote,
})

out_path = os.path.join(output_dir, "prediction_results.tsv")
df_out.to_csv(out_path, sep="\t", index=False)
print(f"\n结果已保存至: {out_path}")
print(f"预测为1: {preds_final.sum()}  预测为0: {(preds_final==0).sum()}")

