import os
import random
import numpy as np
import torch
from faker import Faker


def seed_everything(seed: int) -> None:
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	Faker.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	if torch.backends.cudnn.is_available():
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def print_space():
	print("\n" + "-"*100 + "\n")


def print_header(header):
	print(f"\n### {header}\n")


def nvidia_smi():
	os.system('bash -lc "module load cuda && module load nvidia && nvidia-smi"')
	# os.system('bash -lc "nvidia-smi"')


def check_cuda_driver_consistency():
	nvidia_smi()
	try:
		if not torch.cuda.is_available():
			print("CUDA not available — PyTorch reports no GPUs accessible.")
			return False
		return True

	except Exception as e:
		err = str(e)
		print(f"Unexpected CUDA error:\n{err}")
		return False


def warm_up_mamba_kernels(model, tokenizer):
	"""Run a tiny throwaway generation to warm up mamba_ssm's Triton kernels.

	The Triton autotuner used by mamba_ssm's chunk-scan / conv1d kernels can pick a
	kernel config that triggers a CUDA "illegal memory access" the very first time
	it JIT-compiles for a particular (batch_size, seqlen) shape — this is a known,
	probabilistic upstream issue (see state-spaces/mamba issues #503, #686, #732).
	Priming the kernels on an innocuous, tiny shape first avoids hitting that
	failure mode on the real, much larger, shapes used by the experiments.
	Uses batch size 1 and cg=False to match the sound Mamba generate path used
	by evaluation and gate patching.
	"""
	if type(model).__name__ != "MambaLMHeadModel":
		return
	device = next(model.parameters()).device
	vocab_size = model.config.vocab_size
	dummy_input_ids = torch.randint(0, vocab_size, (1, 8), dtype=torch.long, device=device)
	with torch.inference_mode():
		model.generate(
			input_ids=dummy_input_ids,
			max_length=dummy_input_ids.shape[1] + 4,
			cg=False,
			eos_token_id=tokenizer.eos_token_id,
		)
	if torch.cuda.is_available():
		torch.cuda.synchronize()
		torch.cuda.empty_cache()
