from pathlib import Path

from huggingface_hub import hf_hub_download

# my_edit_reward.py → EditReward/ → repo root (ImgEdit_BalModality)
_REPO = Path(__file__).resolve().parent.parent
_SUB = Path(__file__).resolve().parent

_HF_REPO = "TIGER-Lab/EditReward-MiMo-VL-7B-SFT-2508"
_CKPT_SLUG = "TIGER-Lab__EditReward-MiMo-VL-7B-SFT-2508"
_LOCAL_CKPT = _REPO / ".checkpoints" / _CKPT_SLUG

path = hf_hub_download(
    repo_id=_HF_REPO,
    filename="model.safetensors",
    local_dir=str(_LOCAL_CKPT),
)
print(path)

from src import EditRewardInferencer

cache_dir = str(_REPO / ".cache")
checkpoint_path = str(_LOCAL_CKPT)
config_path = str(_SUB / "src" / "config" / "EditReward-MiMo-VL-7B-SFT-2508.yaml")

inferencer = EditRewardInferencer(config_path, checkpoint_path, cache_dir=cache_dir)

_omni = _REPO / "datasets" / "OmniEdit"
_dist = _REPO / "dist" / "model1_flux_kontext" / "OmniEdit_Result"
_stems = ("task_attr_mod_color_4", "task_attr_mod_color_5")

image_src = [str(_omni / "source" / f"{s}.png") for s in _stems]
image_paths = [str(_dist / f"{s}.png") for s in _stems]
prompts = [
    (_omni / "instruction" / f"{s}.txt").read_text(encoding="utf-8").strip()
    for s in _stems
]

rewards = inferencer.reward(prompts, image_src, image_paths)
print(rewards[0][0].item())
print(rewards[1][0].item())
