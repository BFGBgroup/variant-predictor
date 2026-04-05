# prepare_database.py
# Build the variant feature database from raw annotation files.
#
# Usage:
#   python prepare_database.py \
#       --countdist  /path/to/dbsnp_features_countdist.csv \
#       --enformer   /path/to/enformer_dbsnp_intersect.csv \
#       --favor      /path/to/favor_dbsnp_features_cleaned.csv \
#       --output_dir /path/to/output/

import argparse
import os
import pandas as pd


def check_duplicates(name, df):
    n_total  = len(df)
    n_unique = df["rs_id"].nunique()
    n_dup    = n_total - n_unique
    print(f"  {name}: {n_total} rows, {n_unique} unique rs_id, {n_dup} duplicate(s)")
    if n_dup > 0:
        dup_ids = df["rs_id"][df["rs_id"].duplicated()].unique()
        print(f"    Example duplicates: {list(dup_ids[:5])}")


def main():
    parser = argparse.ArgumentParser(
        description="Build variant feature database from raw dbSNP annotation files."
    )
    parser.add_argument("--countdist",  required=True,
                        help="Path to dbsnp_features_countdist.csv")
    parser.add_argument("--enformer",   required=True,
                        help="Path to enformer_dbsnp_intersect.csv")
    parser.add_argument("--favor",      required=True,
                        help="Path to favor_dbsnp_features_cleaned.csv")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write output .tsv files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading files...")
    df_countdist = pd.read_csv(args.countdist)
    df_enformer  = pd.read_csv(args.enformer)
    df_favor     = pd.read_csv(args.favor)

    print("\nDuplicate check:")
    check_duplicates("countdist", df_countdist)
    check_duplicates("enformer",  df_enformer)
    check_duplicates("favor",     df_favor)

    common_ids = (set(df_countdist["rs_id"])
                  & set(df_enformer["rs_id"])
                  & set(df_favor["rs_id"]))
    print(f"\nCommon variants across all three files: {len(common_ids)}")

    df_countdist = df_countdist[df_countdist["rs_id"].isin(common_ids)]
    df_enformer  = df_enformer[df_enformer["rs_id"].isin(common_ids)]
    df_favor     = df_favor[df_favor["rs_id"].isin(common_ids)]

    out_enformer = os.path.join(args.output_dir, "enformer_dbsnp_intersect_common.tsv")
    out_favor    = os.path.join(args.output_dir, "favor_dbsnp_features_common.tsv")
    df_enformer.to_csv(out_enformer, sep="\t", index=False)
    df_favor.to_csv(out_favor,       sep="\t", index=False)
    print(f"\n  Saved: {out_enformer}")
    print(f"  Saved: {out_favor}")

    count_cols = ["rs_id", "tf_count", "accessible_chromatin_count", "histone_count"]
    dist_cols  = ["rs_id", "tf_dist",  "accessible_chromatin_dist",  "histone_dist"]

    out_count = os.path.join(args.output_dir, "dbsnp_features_count_common.tsv")
    out_dist  = os.path.join(args.output_dir, "dbsnp_features_dist_common.tsv")
    df_countdist[count_cols].to_csv(out_count, sep="\t", index=False)
    df_countdist[dist_cols].to_csv(out_dist,   sep="\t", index=False)
    print(f"  Saved: {out_count}")
    print(f"  Saved: {out_dist}")

    print(f"\nDone. All output files written to: {args.output_dir}")


if __name__ == "__main__":
    main()
