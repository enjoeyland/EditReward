import os
import csv
import time
import torch
from pathlib import Path
from PIL import Image

# EditReward import
from src import EditRewardInferencer


def is_image_file(filename: str) -> bool:
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ext = os.path.splitext(filename)[1].lower()
    return ext in valid_exts


def safe_read_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text


def verify_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def parse_reward_values(reward):
    if isinstance(reward, torch.Tensor):
        return reward.detach().float().cpu().flatten().tolist()
    if isinstance(reward, (list, tuple)):
        return [float(x) for x in reward]
    return [float(reward)]


def compute_final_score(reward_vals):
    if len(reward_vals) == 0:
        return None
    return float(reward_vals[0])


def append_csv_row(writer, file_obj, row_dict):
    writer.writerow(row_dict)
    file_obj.flush()


def main():
    # =========================================================
    # [1] Path settings (aligned with my_edit_reward.py)
    # =========================================================
    _REPO = Path(__file__).resolve().parent.parent
    _SUB = Path(__file__).resolve().parent
    _CKPT_SLUG = "TIGER-Lab__EditReward-MiMo-VL-7B-SFT-2508"
    _LOCAL_CKPT = _REPO / ".checkpoints" / _CKPT_SLUG

    base_folder = str(_REPO / "datasets" / "OmniEdit")
    source_image_folder = os.path.join(base_folder, "source")
    prompt_folder = os.path.join(base_folder, "instruction")
    result_folder = str(_REPO / "dist" / "model1_flux_kontext" / "OmniEdit_Result")

    config_path = str(_SUB / "src" / "config" / "EditReward-MiMo-VL-7B-SFT-2508.yaml")
    checkpoint_path = str(_LOCAL_CKPT)
    cache_dir = str(_REPO / ".cache")

    output_dir = os.path.join(result_folder, "editreward_eval")
    os.makedirs(output_dir, exist_ok=True)

    result_csv_path = os.path.join(output_dir, "editreward_scores.csv")
    fail_csv_path = os.path.join(output_dir, "editreward_failed.csv")
    summary_txt_path = os.path.join(output_dir, "summary.txt")

    # =========================================================
    # [2] Runtime options
    # =========================================================
    batch_size = 4
    print_every_scan = 500
    print_every_batch = 10
    print_sample_preview = True

    # =========================================================
    # [3] Prepare output CSVs first
    # =========================================================
    score_fieldnames = [
        "file",
        "name",
        "prompt",
        "source_path",
        "result_path",
        "reward_0",
        "reward_1",
        "reward_all",
        "final_score",
    ]
    fail_fieldnames = [
        "file",
        "reason",
    ]

    # 기존 파일 덮어쓰기
    with open(result_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=score_fieldnames)
        writer.writeheader()

    with open(fail_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fail_fieldnames)
        writer.writeheader()

    # =========================================================
    # [4] Load inferencer
    # =========================================================
    print("=" * 100)
    print("[1/5] Loading EditReward inferencer...")
    print(f"config_path     : {config_path}")
    print(f"checkpoint_path : {checkpoint_path}")
    print("=" * 100)

    inferencer = EditRewardInferencer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device="cuda",
        reward_dim="overall_detail",
        rm_head_type="ranknet_multi_head",
        cache_dir=cache_dir,
    )

    # =========================================================
    # [5] Gather valid samples
    # =========================================================
    print("=" * 100)
    print("[2/5] Scanning result folder...")
    print(f"result_folder : {result_folder}")
    print("=" * 100)

    files = sorted(os.listdir(result_folder))
    samples = []

    scanned_image_count = 0
    failed_scan_count = 0

    # scan 단계 실패도 즉시 기록
    with open(fail_csv_path, "a", newline="", encoding="utf-8-sig") as fail_f:
        fail_writer = csv.DictWriter(fail_f, fieldnames=fail_fieldnames)

        for file in files:
            result_path = os.path.join(result_folder, file)

            if os.path.isdir(result_path):
                continue
            if not is_image_file(file):
                continue

            scanned_image_count += 1
            if scanned_image_count % print_every_scan == 0:
                print(
                    f"[scan] checked={scanned_image_count} | "
                    f"valid={len(samples)} | failed={failed_scan_count}"
                )

            name, _ = os.path.splitext(file)
            source_path = os.path.join(source_image_folder, file)
            prompt_path = os.path.join(prompt_folder, f"{name}.txt")

            if not os.path.exists(source_path):
                append_csv_row(fail_writer, fail_f, {
                    "file": file,
                    "reason": f"missing source image: {source_path}"
                })
                failed_scan_count += 1
                continue

            if not os.path.exists(prompt_path):
                append_csv_row(fail_writer, fail_f, {
                    "file": file,
                    "reason": f"missing prompt file: {prompt_path}"
                })
                failed_scan_count += 1
                continue

            if not verify_image(source_path):
                append_csv_row(fail_writer, fail_f, {
                    "file": file,
                    "reason": f"corrupted source image: {source_path}"
                })
                failed_scan_count += 1
                continue

            if not verify_image(result_path):
                append_csv_row(fail_writer, fail_f, {
                    "file": file,
                    "reason": f"corrupted result image: {result_path}"
                })
                failed_scan_count += 1
                continue

            try:
                prompt = safe_read_prompt(prompt_path)
            except Exception as e:
                append_csv_row(fail_writer, fail_f, {
                    "file": file,
                    "reason": f"prompt read error: {str(e)}"
                })
                failed_scan_count += 1
                continue

            if prompt == "":
                append_csv_row(fail_writer, fail_f, {
                    "file": file,
                    "reason": "empty prompt"
                })
                failed_scan_count += 1
                continue

            samples.append({
                "file": file,
                "name": name,
                "prompt": prompt,
                "source_path": source_path,
                "result_path": result_path,
            })

    print(f"[scan done] total checked : {scanned_image_count}")
    print(f"[scan done] valid         : {len(samples)}")
    print(f"[scan done] failed        : {failed_scan_count}")

    if len(samples) == 0:
        print("[STOP] No valid samples found.")
        return

    # =========================================================
    # [6] Evaluate in batches, save each sample immediately
    # =========================================================
    print("=" * 100)
    print("[3/5] Starting evaluation...")
    print(f"batch_size   : {batch_size}")
    print(f"num_samples  : {len(samples)}")
    total_batches = (len(samples) + batch_size - 1) // batch_size
    print(f"num_batches  : {total_batches}")
    print("=" * 100)

    start_time_all = time.time()

    success_count = 0
    failed_eval_count = 0
    final_score_sum = 0.0
    final_score_min = None
    final_score_max = None

    with open(result_csv_path, "a", newline="", encoding="utf-8-sig") as score_f, \
         open(fail_csv_path, "a", newline="", encoding="utf-8-sig") as fail_f:

        score_writer = csv.DictWriter(score_f, fieldnames=score_fieldnames)
        fail_writer = csv.DictWriter(fail_f, fieldnames=fail_fieldnames)

        for batch_idx, start_idx in enumerate(range(0, len(samples), batch_size), start=1):
            end_idx = min(start_idx + batch_size, len(samples))
            batch = samples[start_idx:end_idx]

            prompts = [item["prompt"] for item in batch]
            image_src = [item["source_path"] for item in batch]
            image_paths = [item["result_path"] for item in batch]

            batch_start_time = time.time()

            if batch_idx == 1 or batch_idx % print_every_batch == 0:
                print("-" * 100)
                print(f"[batch {batch_idx}/{total_batches}] start")
                print(f"sample range  : {start_idx} ~ {end_idx - 1}")
                print(f"example file  : {batch[0]['file']}")
                print(f"example prompt: {batch[0]['prompt'][:120]}")
                print("-" * 100)

            try:
                with torch.no_grad():
                    rewards = inferencer.reward(
                        prompts=prompts,
                        image_src=image_src,
                        image_paths=image_paths,
                    )

                print(f"[batch {batch_idx}] rewards.shape = {getattr(rewards, 'shape', 'N/A')}")

                for sample_idx, (item, reward) in enumerate(zip(batch, rewards)):
                    try:
                        reward_vals = parse_reward_values(reward)

                        reward_0 = reward_vals[0] if len(reward_vals) > 0 else None
                        reward_1 = reward_vals[1] if len(reward_vals) > 1 else None
                        final_score = compute_final_score(reward_vals)

                        row = {
                            "file": item["file"],
                            "name": item["name"],
                            "prompt": item["prompt"],
                            "source_path": item["source_path"],
                            "result_path": item["result_path"],
                            "reward_0": reward_0,
                            "reward_1": reward_1,
                            "reward_all": ",".join([f"{x:.6f}" for x in reward_vals]),
                            "final_score": final_score,
                        }

                        append_csv_row(score_writer, score_f, row)
                        success_count += 1

                        if final_score is not None:
                            final_score_sum += final_score
                            if final_score_min is None or final_score < final_score_min:
                                final_score_min = final_score
                            if final_score_max is None or final_score > final_score_max:
                                final_score_max = final_score

                        if print_sample_preview and (batch_idx == 1 or batch_idx % print_every_batch == 0):
                            print(
                                f"[sample {sample_idx}] "
                                f"file={item['file']} | "
                                f"reward_0={reward_0} | "
                                f"reward_1={reward_1} | "
                                f"final_score={final_score}"
                            )

                    except Exception as e:
                        append_csv_row(fail_writer, fail_f, {
                            "file": item["file"],
                            "reason": f"score parse error: {str(e)}"
                        })
                        failed_eval_count += 1

                batch_elapsed = time.time() - batch_start_time

                if batch_idx == 1 or batch_idx % print_every_batch == 0 or batch_idx == total_batches:
                    elapsed_all = time.time() - start_time_all
                    avg_sec_per_batch = elapsed_all / batch_idx
                    remain_batches = total_batches - batch_idx
                    eta_sec = avg_sec_per_batch * remain_batches

                    print(f"[progress] batch done      : {batch_idx}/{total_batches}")
                    print(f"[progress] processed        : {end_idx}/{len(samples)}")
                    print(f"[progress] success rows     : {success_count}")
                    print(f"[progress] failed rows      : {failed_scan_count + failed_eval_count}")
                    print(f"[progress] batch time       : {batch_elapsed:.2f}s")
                    print(f"[progress] elapsed total    : {elapsed_all/60:.2f} min")
                    print(f"[progress] ETA             : {eta_sec/60:.2f} min")
                    print(f"[progress] score csv path   : {result_csv_path}")

            except Exception as e:
                batch_files = [item["file"] for item in batch]
                print(f"[Batch Error] batch={batch_idx}/{total_batches}")
                print(f"[Batch Error] files={batch_files}")
                print(f"[Batch Error] reason={str(e)}")

                for item in batch:
                    append_csv_row(fail_writer, fail_f, {
                        "file": item["file"],
                        "reason": f"inference error: {str(e)}"
                    })
                    failed_eval_count += 1

    # =========================================================
    # [7] Save summary
    # =========================================================
    print("=" * 100)
    print("[4/5] Writing summary...")
    print("=" * 100)

    if success_count > 0:
        avg_score = final_score_sum / success_count
    else:
        avg_score = None

    num_failed = failed_scan_count + failed_eval_count

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("EditReward Evaluation Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"result_folder   : {result_folder}\n")
        f.write(f"num_success     : {success_count}\n")
        f.write(f"num_failed      : {num_failed}\n")
        f.write(f"batch_size      : {batch_size}\n")
        f.write(f"config_path     : {config_path}\n")
        f.write(f"checkpoint_path : {checkpoint_path}\n")
        f.write(f"score_csv       : {result_csv_path}\n")
        f.write(f"fail_csv        : {fail_csv_path}\n")
        f.write("\n")
        f.write(f"avg_final_score : {avg_score}\n")
        f.write(f"min_final_score : {final_score_min}\n")
        f.write(f"max_final_score : {final_score_max}\n")

    print("=" * 100)
    print("[5/5] Done.")
    print(f"Saved score csv : {result_csv_path}")
    print(f"Saved fail csv  : {fail_csv_path}")
    print(f"Saved summary   : {summary_txt_path}")
    print(f"Success         : {success_count}")
    print(f"Failed          : {num_failed}")
    if avg_score is not None:
        print(f"Average score   : {avg_score:.6f}")
        print(f"Min score       : {final_score_min:.6f}")
        print(f"Max score       : {final_score_max:.6f}")
    print("=" * 100)


if __name__ == "__main__":
    main()