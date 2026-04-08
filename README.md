```
# Variant Predictor

A command-line tool for predicting the transcriptional regulatory activity of genetic variants.  
The model is trained on large-scale MPRA (Massively Parallel Reporter Assay) data using a 
pre-trained multi-modal deep learning framework built on IntegrAO (https://github.com/bowang-lab/IntegrAO).  

Given a list of dbSNP rs IDs, the tool extracts four types of genomic features and returns ensemble predictions across 5 cross-validation folds.

---

## How It Works

Four feature modalities:

| Modality  | Description |
|-----------|-------------|
| Enformer  | Deep learning sequence embeddings (20 PCs) |
| FAVOR     | Epigenetic annotation scores (7 features) |
| Count     | TF / chromatin / histone overlap counts |
| Distance  | Distance to nearest TF / chromatin / histone peak |

---

## Requirements

- Python >= 3.8  
- IntegrAO installed (https://github.com/bowang-lab/IntegrAO)  
- CUDA-capable GPU recommended (CPU inference supported but slower)  

---

## Installation

# 1. Clone this repository
git clone https://github.com/BFGBgroup/variant-predictor.git
cd variant-predictor

# 2. Create and activate a conda environment
conda create -n variant_pred python=3.10
conda activate variant_pred

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install IntegrAO
git clone https://github.com/bowang-lab/IntegrAO.git IntegrAO
cd IntegrAO && pip install -e . && cd ..

---

## Feature Database

Download the feature database (~5.3 GB) from Zenodo:

https://doi.org/10.5281/zenodo.19447280

Place all four files in the `feature_db/` directory:

variant-predictor/
├── feature_db/
│   ├── enformer_dbsnp_intersect_common.tsv
│   ├── favor_dbsnp_features_common.tsv
│   ├── dbsnp_features_count_common.tsv
│   └── dbsnp_features_dist_common.tsv

---

## Models

Put the 5 trained model files in the `models/` directory:

variant-predictor/
├── models/
│   ├── model_fold1.pth
│   ├── model_fold2.pth
│   ├── model_fold3.pth
│   ├── model_fold4.pth
│   └── model_fold5.pth

---

## Run Predictions

Input format: one rs ID per line:

rs367896724
rs555500075
rs376342519

Run the script:

python predict_variants.py example/example_variants.txt [output_dir]

- If `[output_dir]` is omitted, results are saved in the same directory as the input file.  
- The script automatically detects `feature_db/` and `models/` directories.  

---

## Output

Results are saved to `prediction_results.tsv`:

| Column      | Description |
|-------------|-------------|
| Variant     | rs ID |
| probs_mean  | Mean P(functional) across 5 folds (0-1) |
| preds_final | Final label: 1 = functional, 0 = non-functional |
| preds_vote  | Number of folds predicting class 1 (0-5) |

---

## Reference

- IntegrAO: https://github.com/bowang-lab/IntegrAO  
- Enformer: Avsec et al., Nature Methods, 2021  
- FAVOR: Zhou et al., Nature Genetics, 2022
```
