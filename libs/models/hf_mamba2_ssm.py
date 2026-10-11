import json
import os

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


CACHE_DIR = os.environ.get("HF_CACHE_DIR") or None

# CLI model id -> HF Hub repo id. Every entry is loaded through the same
# HF-config/weights -> mamba_ssm.MambaLMHeadModel remapping below. Checkpoints
# may be a single model.safetensors (AntonV) or sharded with an index
# (Codestral); embeddings may be tied or untied. Registration in
# libs/models/models.py is still required for each new CLI id.
HF_MAMBA2_REPOS = {
	"mamba_codestral": "mistralai/Mamba-Codestral-7B-v0.1",
}


def _cache_kwargs() -> dict:
	return {"cache_dir": CACHE_DIR} if CACHE_DIR else {}


def _hf_config_to_mamba_config(hf_config) -> MambaConfig:
	"""Translate a transformers Mamba2Config into mamba_ssm's MambaConfig.

	Field-name mapping (HF -> mamba_ssm): hidden_size -> d_model,
	num_hidden_layers -> n_layer, head_dim -> ssm_cfg['headdim'],
	state_size -> ssm_cfg['d_state'], n_groups -> ssm_cfg['ngroups'].
	pad_vocab_size_multiple=1 keeps vocab_size exactly as published by HF
	(it's already whatever padding the original checkpoint needed).
	tie_embeddings follows HF's tie_word_embeddings (True for AntonV, False
	for Codestral, which ships a separate lm_head.weight).
	"""
	return MambaConfig(
		d_model=hf_config.hidden_size,
		n_layer=hf_config.num_hidden_layers,
		vocab_size=hf_config.vocab_size,
		ssm_cfg={
			'layer': 'Mamba2',
			'expand': hf_config.expand,
			'headdim': hf_config.head_dim,
			'd_state': hf_config.state_size,
			'ngroups': hf_config.n_groups,
		},
		rms_norm=hf_config.rms_norm,
		residual_in_fp32=hf_config.residual_in_fp32,
		fused_add_norm=True,
		pad_vocab_size_multiple=1,
		tie_embeddings=hf_config.tie_word_embeddings,
	)


def _map_hf_to_mamba_ssm(state_dict: dict) -> dict:
	"""Rename HF Mamba2 checkpoint keys to mamba_ssm's MambaLMHeadModel naming.

	Only the top-level embedding key differs ('embeddings' -> 'embedding');
	backbone.norm_f, backbone.layers.*.norm and backbone.layers.*.mixer.*
	(A_log, D, conv1d, dt_bias, in_proj, norm, out_proj) already match
	mamba_ssm's own naming, since both implementations trace back to the
	same original mamba_ssm module. lm_head.weight is left as-is when present
	(untied checkpoints such as Codestral). Tied checkpoints (AntonV) omit
	lm_head.weight: MambaLMHeadModel.tie_weights() aliases lm_head.weight to
	backbone.embedding.weight in __init__, so loading only the embedding key
	(with strict=False) updates both.
	"""
	renames = {
		'backbone.embeddings.weight': 'backbone.embedding.weight',
	}
	return {renames.get(k, k): v.to(torch.bfloat16) for k, v in state_dict.items()}


def _load_hf_state_dict(repo_id: str) -> dict:
	"""Load a HF safetensors checkpoint, supporting single-file and sharded layouts.

	Prefers model.safetensors.index.json + shards when present (Codestral);
	falls back to a single model.safetensors (AntonV).

	Weights are resolved from HuggingFace's cache (override with HF_CACHE_DIR /
	HF_HOME). Offline compute nodes should set HF_HUB_OFFLINE=1 after a
	prefetch (`huggingface-cli download` on a node with internet access).
	"""
	try:
		index_path = hf_hub_download(
			repo_id=repo_id,
			filename="model.safetensors.index.json",
			**_cache_kwargs(),
		)
	except (EntryNotFoundError, LocalEntryNotFoundError):
		weights_path = hf_hub_download(
			repo_id=repo_id,
			filename="model.safetensors",
			**_cache_kwargs(),
		)
		return load_file(weights_path)

	with open(index_path) as f:
		index = json.load(f)
	shard_files = sorted(set(index["weight_map"].values()))
	state_dict = {}
	for shard in shard_files:
		print(f"  Loading shard {shard}...")
		shard_path = hf_hub_download(
			repo_id=repo_id,
			filename=shard,
			**_cache_kwargs(),
		)
		state_dict.update(load_file(shard_path))
	return state_dict


def get_model(model_id: str) -> MambaLMHeadModel:
	repo_id = HF_MAMBA2_REPOS[model_id]

	print(f"Loading config for {repo_id}...")
	hf_config = AutoConfig.from_pretrained(repo_id, **_cache_kwargs())
	config = _hf_config_to_mamba_config(hf_config)

	print("Loading weights...")
	mapped_state_dict = _map_hf_to_mamba_ssm(_load_hf_state_dict(repo_id))

	print("Initializing model on GPU...")
	model = MambaLMHeadModel(config, device="cuda", dtype=torch.bfloat16)

	# strict=False: tied checkpoints have no standalone lm_head.weight (see
	# _map_hf_to_mamba_ssm docstring above). Untied checkpoints (Codestral)
	# include lm_head.weight and should load cleanly either way.
	msg = model.load_state_dict(mapped_state_dict, strict=False)
	print(f"Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
	if msg.missing_keys:
		print(f"  missing (sample): {msg.missing_keys[:5]}")
	if msg.unexpected_keys:
		print(f"  unexpected (sample): {msg.unexpected_keys[:5]}")

	return model


def get_tokenizer(model_id: str):
	repo_id = HF_MAMBA2_REPOS[model_id]
	return AutoTokenizer.from_pretrained(repo_id, **_cache_kwargs())
