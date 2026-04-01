from pathlib import Path
from typing import Any, List

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def _as_train_samples(dataset_obj: Any) -> List[dict]:
    if isinstance(dataset_obj, DatasetDict):
        if "train" in dataset_obj:
            split = dataset_obj["train"]
        else:
            first_key = next(iter(dataset_obj.keys()))
            split = dataset_obj[first_key]
    elif isinstance(dataset_obj, Dataset):
        split = dataset_obj
    else:
        raise TypeError(f"Unsupported dataset object type: {type(dataset_obj)}")
    return [split[i] for i in range(len(split))]


def load_editreward_samples(dataset_name_or_path: str) -> List[dict]:
    """
    Load EditReward-Bench samples from either:
    1. A Hugging Face dataset name like `TIGER-Lab/EditReward-Bench`
    2. A local `datasets.save_to_disk(...)` directory
    3. A local dataset script / local HF dataset directory accepted by `load_dataset`
    """
    source = Path(dataset_name_or_path).expanduser()

    if source.exists():
        try:
            dataset_obj = load_from_disk(str(source))
            print(f"✅ Loaded local dataset with load_from_disk: {source}")
            return _as_train_samples(dataset_obj)
        except Exception:
            dataset_obj = load_dataset(str(source))
            print(f"✅ Loaded local dataset with load_dataset: {source}")
            return _as_train_samples(dataset_obj)

    dataset_obj = load_dataset(dataset_name_or_path)
    print(f"✅ Loaded remote dataset: {dataset_name_or_path}")
    return _as_train_samples(dataset_obj)
