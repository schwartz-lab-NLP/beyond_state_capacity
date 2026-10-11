from libs.models.nvidia_mamba import get_model as get_nvidia_mamba_model, get_tokenizer as get_nvidia_mamba_tokenizer
from libs.models.hf_mamba2_ssm import (
	HF_MAMBA2_REPOS,
	get_model as get_hf_mamba2_model,
	get_tokenizer as get_hf_mamba2_tokenizer,
)


MAMBA_MODELS = ["mamba_nvidia", "mamba_codestral"]
ALL_MODELS = MAMBA_MODELS


def get_tokenizer(args):
	if args.model == "mamba_nvidia":
		tokenizer = get_nvidia_mamba_tokenizer()
	elif args.model in HF_MAMBA2_REPOS:
		tokenizer = get_hf_mamba2_tokenizer(args.model)
	else:
		raise ValueError(f"Model {args.model} not supported")

	# Left-padding keeps generated continuations at the same offset for every
	# sample in a batch.
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "left"
	return tokenizer


def get_model(args):
	if args.model == "mamba_nvidia":
		return get_nvidia_mamba_model()
	elif args.model in HF_MAMBA2_REPOS:
		return get_hf_mamba2_model(args.model)
	else:
		raise ValueError(f"Model {args.model} not supported")


def print_model(model):
	param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
	if param_count < 1_000:
		size_str = f"{param_count}"
	elif param_count < 1_000_000:
		size_str = f"{param_count / 1_000:.2f}K"
	elif param_count < 1_000_000_000:
		size_str = f"{param_count / 1_000_000:.2f}M"
	elif param_count < 1_000_000_000_000:
		size_str = f"{param_count / 1_000_000_000:.2f}B"
	else:
		size_str = f"{param_count / 1_000_000_000_000:.2f}T"

	if size_str.endswith(".00K") or size_str.endswith(".00M") or size_str.endswith(".00B") or size_str.endswith(".00T"):
		size_str = size_str.replace(".00", "")

	print(f"Number of parameters of the model: {size_str}\n")
	print(model)
