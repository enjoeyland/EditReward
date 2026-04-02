import os
import sys
import time
import torch
from pathlib import Path
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parent.parent
_INF = _REPO / "Inferences"
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))

from datamodules.omniedit import OmniEditLocalDataset
from datamodules.saved_edits import PairedPredGtDataset
from utils.sample_metrics_sqlite import SampleMetricsStore

from src import EditRewardInferencer

EDITREWARD_METRIC_FOLLOW = "editreward/following"
EDITREWARD_METRIC_QUALITY = "editreward/quality"

def collate_fn(batch):
    return {
        "sample_id": [b["sample_id"] for b in batch],
        "prompt": [Path(b["prompt_path"]).read_text(encoding="utf-8").strip() for b in batch],
        "source_path": [b["source_path"] for b in batch],
        "result_path": [b["edited_path"] for b in batch],
    }

def main():
    # =========================================================
    # [1] Path settings (aligned with my_edit_reward.py)
    # =========================================================
    _SUB = Path(__file__).resolve().parent
    _CKPT_SLUG = "TIGER-Lab__EditReward-MiMo-VL-7B-SFT-2508"
    _LOCAL_CKPT = _REPO / ".checkpoints" / _CKPT_SLUG

    base_folder = str(_REPO / "datasets" / "OmniEdit")
    result_folder = str(_REPO / "dist" / "model1_flux_kontext" / "OmniEdit_Result")

    config_path = str(_SUB / "src" / "config" / "EditReward-MiMo-VL-7B-SFT-2508.yaml")
    checkpoint_path = str(_LOCAL_CKPT)
    cache_dir = str(_REPO / ".cache")

    # =========================================================
    # [2] Runtime options
    # =========================================================
    batch_size = 1
    num_workers = 4

    # =========================================================
    # [4] Gather valid samples (OmniEdit GT + PairedPredGtDataset, as_paths=True)
    # =========================================================
    print("=" * 100)
    print("[1/5] Pairing GT (OmniEdit) with prediction PNGs...")
    print(f"base_folder   : {base_folder}")
    print(f"result_folder : {result_folder}")
    print("=" * 100)

    gt_ds = OmniEditLocalDataset(base_folder, 0, 20000, as_paths=True)
    paired = PairedPredGtDataset(gt_ds, result_folder, as_paths=True)
    assert len(paired) > 0, "No paired samples found"

    dataloader = DataLoader(paired, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # =========================================================
    # [5] Load inferencer
    # =========================================================
    print("=" * 100)
    print("[2/5] Loading EditReward inferencer...")
    print(f"config_path     : {config_path}")
    print(f"checkpoint_path : {checkpoint_path}")
    print("=" * 100)

    inferencer = EditRewardInferencer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        reward_dim="overall_detail",
        rm_head_type="ranknet_multi_head",
        cache_dir=cache_dir,
    )

    db_path = str(_REPO / "dist" / "sample_metrics.sqlite")
    metrics_run_id = time.strftime("editreward_eval_%Y%m%d_%H%M")
    metrics_store = SampleMetricsStore(
        db_path=db_path,
        run_id=metrics_run_id,
        task="edit_score",
        model="flux_kontext",
        dataset="omniedit",
    )
    print(f"Sample metrics DB: {db_path} (run_id={metrics_run_id})")

    # =========================================================
    # [6] Evaluate in batches, save each sample immediately
    # =========================================================
    print("=" * 100)
    print("[3/5] Starting evaluation...")
    print("=" * 100)

    n_batches = len(dataloader)
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold magenta]{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TextColumn("[cyan]follow={task.fields[follow]:.4f}"),
        TextColumn("[cyan]quality={task.fields[quality]:.4f}"),
    ) as progress:
        task_id = progress.add_task("Evaluating", total=n_batches, follow=0.0, quality=0.0)
        for batch in dataloader:
            with torch.no_grad():
                rewards = inferencer.reward(
                    prompts=batch["prompt"],
                    image_src=batch["source_path"],
                    image_paths=batch["result_path"],
                )
            r = rewards.detach().float().cpu()

            for sid, row in zip(batch["sample_id"], r):
                metrics_store.upsert_many(
                    {
                        "sample_id": str(sid),
                        EDITREWARD_METRIC_FOLLOW: float(row[0]),
                        EDITREWARD_METRIC_QUALITY: float(row[1]),
                    }
                )
            
            follow_avg = float(r[:, 0].mean())
            quality_avg = float(r[:, 1].mean())
            progress.update(task_id, advance=1, follow=follow_avg, quality=quality_avg)
    

if __name__ == "__main__":
    main()