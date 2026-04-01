import os
import pandas as pd


def main():
    dino_csv_file = "/scratch2/[SC_LAB]/dreamyou070/Neurips_MultimodalBiased/Step1_DataCuration/model2_qwen_2509/OmniEdit_concat_save/dino.csv"
    reward_csv_file = "/scratch2/[SC_LAB]/dreamyou070/Neurips_MultimodalBiased/Step1_DataCuration/model2_qwen_2509/OmniEdit_concat_save/all_sorted_scores1.csv"

    save_dir = "/scratch2/[SC_LAB]/dreamyou070/Neurips_MultimodalBiased/Step1_DataCuration/model2_qwen_2509/OmniEdit_concat_save"
    os.makedirs(save_dir, exist_ok=True)

    merged_csv_path = os.path.join(save_dir, "reward_dino_merged.csv")
    summary_csv_path = os.path.join(save_dir, "reward_dino_category_summary.csv")

    # --------------------------------------------------
    # 1. load
    # --------------------------------------------------
    dino_df = pd.read_csv(
        dino_csv_file,
        header=None,
        names=["file", "source_path", "result_path", "dino_score"]
    )

    reward_df = pd.read_csv(reward_csv_file)

    # --------------------------------------------------
    # 2. numeric conversion
    # --------------------------------------------------
    dino_df["dino_score"] = pd.to_numeric(dino_df["dino_score"], errors="coerce")
    reward_df["final_score"] = pd.to_numeric(reward_df["final_score"], errors="coerce")

    # reward 기준 파일명만 남김
    reward_df["file"] = reward_df["file"].astype(str)
    dino_df["file"] = dino_df["file"].astype(str)

    # --------------------------------------------------
    # 3. merge
    # --------------------------------------------------
    merged = pd.merge(
        reward_df,
        dino_df[["file", "dino_score"]],
        on="file",
        how="left"
    )

    # reward는 있고 dino도 있는 샘플만 분석 대상으로
    valid = merged.dropna(subset=["final_score", "dino_score"]).copy()

    if len(valid) == 0:
        raise ValueError("유효한 final_score + dino_score 샘플이 없습니다.")

    print("===== Basic Info =====")
    print(f"reward rows : {len(reward_df)}")
    print(f"dino rows   : {len(dino_df)}")
    print(f"merged rows : {len(merged)}")
    print(f"valid rows  : {len(valid)}")

    # --------------------------------------------------
    # 4. threshold
    # --------------------------------------------------
    reward_thr = valid["final_score"].median()
    dino_thr = valid["dino_score"].median()

    print("\n===== Thresholds =====")
    print(f"reward median threshold : {reward_thr:.6f}")
    print(f"dino   median threshold : {dino_thr:.6f}")

    # --------------------------------------------------
    # 5. category assignment
    # --------------------------------------------------
    def classify(row):
        reward = row["final_score"]
        dino = row["dino_score"]

        if reward >= reward_thr:
            return "optimized"
        else:
            if dino >= dino_thr:
                return "under_edit"
            else:
                return "over_edit"

    valid["category"] = valid.apply(classify, axis=1)

    # --------------------------------------------------
    # 6. summary
    # --------------------------------------------------
    category_order = ["optimized", "under_edit", "over_edit"]
    counts = valid["category"].value_counts().reindex(category_order, fill_value=0)
    ratios = counts / counts.sum()

    summary_df = pd.DataFrame({
        "category": category_order,
        "count": [counts[c] for c in category_order],
        "ratio": [ratios[c] for c in category_order],
    })

    print("\n===== Category Summary =====")
    print(summary_df)

    # --------------------------------------------------
    # 7. save
    # --------------------------------------------------
    valid.to_csv(merged_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    print("\nSaved files:")
    print(merged_csv_path)
    print(summary_csv_path)


if __name__ == "__main__":
    main()