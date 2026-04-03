import argparse
import time
from pathlib import Path
from typing import cast

import torch
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch.utils.data import DataLoader

from omegaconf import DictConfig, OmegaConf
from huggingface_hub import hf_hub_download

from datamodules import MagicBrushLocalDataset, OmniEditLocalDataset, PairedPredGtDataset
from utils.core import resolve_edit_save_root
from utils.sample_metrics_sqlite import SampleMetricsStore

from src import EditRewardInferencer

_EDIT = Path(__file__).resolve().parent
_REPO = _EDIT.parent
_INF = _REPO / "Inferences"

EDITREWARD_METRIC_FOLLOW = "editreward/following"
EDITREWARD_METRIC_QUALITY = "editreward/quality"

def _config_stems(config_subdir: str) -> list[str]:
    d = _INF / "config" / config_subdir
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.yaml") if p.is_file())


def _load_model_dataset_cfg(model_name: str, dataset_name: str) -> tuple[DictConfig, DictConfig]:
    cfg_dir = _INF / "config"
    model_cfg = cast(DictConfig, OmegaConf.load(cfg_dir / "models" / f"{model_name}.yaml"))
    dataset_cfg = cast(DictConfig, OmegaConf.load(cfg_dir / "datasets" / f"{dataset_name}.yaml"))
    return model_cfg, dataset_cfg


def collate_fn(batch):
    return {
        "sample_id": [b["sample_id"] for b in batch],
        "prompt": [Path(b["prompt_path"]).read_text(encoding="utf-8").strip() for b in batch],
        "source_path": [b["source_path"] for b in batch],
        "result_path": [b["edited_path"] for b in batch],
    }


def main():
    models = _config_stems("models")
    datasets = _config_stems("datasets")
    if not models or not datasets:
        raise SystemExit(f"Missing YAML under {_INF / 'config'}: models={models!r} datasets={datasets!r}")

    p = argparse.ArgumentParser(description="EditReward batch eval on paired GT + inference PNGs (paths from Inferences/config).")
    p.add_argument("--dataset", choices=datasets, default="omniedit", help="Inferences/config/datasets/<name>.yaml")
    p.add_argument("--model", choices=models, default="flux_kontext", help="Inferences/config/models/<name>.yaml")
    p.add_argument("--sample-id-start", type=int, default=0, help="Half-open lower bound [start, end) for dataset filter.")
    p.add_argument("--sample-id-end", type=int, default=20000, help="Half-open upper bound (exclusive).")
    args = p.parse_args()

    model_cfg, dataset_cfg = _load_model_dataset_cfg(args.model, args.dataset)
    base_folder = str((_REPO / str(dataset_cfg.local.root)).resolve())
    result_folder = str(resolve_edit_save_root(_REPO / "dist", model_cfg, dataset_cfg))

    config_path = str(_EDIT / "src" / "config" / "EditReward-MiMo-VL-7B-SFT-2508.yaml")
    checkpoint_path = str(_REPO / ".checkpoints" / "TIGER-Lab__EditReward-MiMo-VL-7B-SFT-2508")
    cache_dir = str(_REPO / ".cache")

    _HF_REPO = "TIGER-Lab/EditReward-MiMo-VL-7B-SFT-2508"
    hf_hub_download(repo_id=_HF_REPO, filename="model.safetensors", local_dir=checkpoint_path)

    batch_size = 1
    num_workers = 4
    sample_id_start = args.sample_id_start
    sample_id_end = args.sample_id_end

    ds_name = str(dataset_cfg.name).lower()
    if ds_name == "omniedit":
        gt_ds = OmniEditLocalDataset(base_folder, sample_id_start, sample_id_end, as_paths=True)
    elif ds_name == "magicbrush":
        gt_ds = MagicBrushLocalDataset(base_folder, sample_id_start, sample_id_end, as_paths=True)
    else:
        raise SystemExit(f"Unsupported dataset.name={dataset_cfg.name!r} (add branch or use omniedit/magicbrush).")

    print("=" * 100)
    print("[1/5] Pairing GT with prediction PNGs...")
    print(f"dataset       : {args.dataset} (root={base_folder}")
    print(f"model         : {args.model} (pred_dir={result_folder})")
    print(f"# of samples  : {len(gt_ds)}")
    print(f"sample_id     : [{sample_id_start}, {sample_id_end})")
    print("=" * 100)

    paired = PairedPredGtDataset(gt_ds, result_folder, as_paths=True)
    assert len(paired) > 0, "No paired samples found"

    dataloader = DataLoader(paired, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    print("=" * 100)
    print("[2/5] Loading EditReward inferencer...")
    print(f"config_path     : {config_path}")
    print(f"checkpoint_path : {checkpoint_path}")
    print(f"cache_dir       : {cache_dir}")
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
        model=str(model_cfg.name),
        dataset=str(dataset_cfg.name),
    )
    print(f"Sample metrics DB: {db_path} (run_id={metrics_run_id})")

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
