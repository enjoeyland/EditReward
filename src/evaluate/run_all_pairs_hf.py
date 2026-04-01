import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


"""
Example:

export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
export QWEN3_5_MODEL_NAME="Qwen3.5-27B"

python run_all_pairs_hf.py \
  --model_name qwen3.5 \
  --dataset_name /path/to/EditReward-Bench \
  --max_workers 16 \
  --results_root outputs/results_qwen35_allpairs_full \
  --overwrite
"""

PAIR_CONFIGS = {
    "2pair": {
        "script": "inference_2pair_hf.py",
        "template": "pairwise_2pair",
    },
    "3pair": {
        "script": "inference_3pair_hf.py",
        "template": "pairwise_3pair",
    },
    "4pair": {
        "script": "inference_4pair_hf.py",
        "template": "pairwise_4pair",
    },
}


def parse_pairs(raw_pairs: str) -> List[str]:
    items = [item.strip() for item in raw_pairs.split(",") if item.strip()]
    invalid = [item for item in items if item not in PAIR_CONFIGS]
    if invalid:
        raise ValueError(f"Unsupported pair types: {invalid}")
    if not items:
        raise ValueError("No pair types specified.")
    return items


def build_command(
    python_executable: str,
    script_path: Path,
    model_name: str,
    dataset_name: str,
    template: str,
    results_dir: Path,
    overwrite: bool,
    max_examples: int | None,
    max_workers: int,
) -> List[str]:
    command = [
        python_executable,
        str(script_path),
        "--model_name",
        model_name,
        "--dataset_name",
        dataset_name,
        "--template",
        template,
        "--results_dir",
        str(results_dir),
        "--max_workers",
        str(max_workers),
    ]
    if max_examples is not None:
        command.extend(["--max_examples", str(max_examples)])
    if overwrite:
        command.extend(["--overwrite", "True"])
    return command


def run_single_command(command: List[str], cwd: Path) -> int:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n[Run] {printable}\n")
    completed = subprocess.run(command, cwd=str(cwd))
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EditReward-Bench evaluation for multiple pair settings with one command."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Registered model name in genaibench/mllm_tools, e.g. qwen3.5",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="TIGER-Lab/EditReward-Bench",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="2pair,3pair,4pair",
        help="Comma-separated pair types to run. Choices: 2pair,3pair,4pair",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results_all_pairs",
        help="Root directory to store per-pair result folders.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap passed through to each pair script.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of workers passed to each pair script.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Forward overwrite=True to each pair script.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Keep running remaining pair scripts even if one fails.",
    )

    args = parser.parse_args()

    selected_pairs = parse_pairs(args.pairs)
    script_dir = Path(__file__).resolve().parent
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = (script_dir / results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    summary: List[Tuple[str, int]] = []
    for pair_name in selected_pairs:
        config = PAIR_CONFIGS[pair_name]
        script_path = script_dir / config["script"]
        results_dir = results_root / pair_name
        results_dir.mkdir(parents=True, exist_ok=True)

        command = build_command(
            python_executable=sys.executable,
            script_path=script_path,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            template=config["template"],
            results_dir=results_dir,
            overwrite=args.overwrite,
            max_examples=args.max_examples,
            max_workers=args.max_workers,
        )
        return_code = run_single_command(command, cwd=script_dir)
        summary.append((pair_name, return_code))
        if return_code != 0 and not args.continue_on_error:
            print(f"[Stop] {pair_name} failed with exit code {return_code}.")
            sys.exit(return_code)

    print("\n[Summary]")
    for pair_name, return_code in summary:
        status = "OK" if return_code == 0 else f"FAILED({return_code})"
        print(f"- {pair_name}: {status}")

    if any(return_code != 0 for _, return_code in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
