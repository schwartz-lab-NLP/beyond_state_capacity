"""Shared Experiment base class used by every paper and exploration experiment.

Provides encoding/decoding and generation helpers that work for both HuggingFace
tokenizers and bare SentencePiece processors, dataset construction, and
output-path layout via libs.paths. Subclasses implement run() and save_plots();
results are written as JSON via save_results().
"""

from abc import ABC, abstractmethod
import json
import os
import re

import torch
import sentencepiece as spm
import numpy as np

from libs.datasets.phonebook import PhonebookDataset
from libs.datasets.profbook import ProfbookDataset
from libs.models.models import TRANSFORMER_MODELS
from libs.paths import experiment_plots_dir, experiment_raw_results_dir


class _NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		return super().default(obj)


TRANSFORMER_MAX_BATCH = 4
MAMBA_MAX_BATCH = 32


class Experiment(ABC):
	def __init__(self, args, model, tokenizer):
		self.args = args
		self.model = model
		self.tokenizer = tokenizer
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.results = {}

	@abstractmethod
	def run(self):
		pass
	
	@property
	def _variant_prefix(self):
		"""String prepended to all output filenames to identify this run's variant.

		Subclasses override this to distinguish outputs produced under different
		argument combinations (e.g. 'norm_', 'cosine_', 'whitespaces_').
		The empty string means no qualifier (default / single-variant experiment).
		"""
		return ""

	@staticmethod
	def _make_results_filename(variant_prefix):
		return f"{variant_prefix}results.json"

	@property
	def _results_filename(self):
		return Experiment._make_results_filename(self._variant_prefix)

	@property
	def _output_path_extra(self) -> tuple[str, ...]:
		"""Extra path segments under raw_results/ and plots/ (created automatically)."""
		return ()

	def save_results(self):
		path = self._get_raw_results_path()
		os.makedirs(path, exist_ok=True)
		with open(os.path.join(path, self._results_filename), "w") as f:
			json.dump(self.results, f, cls=_NumpyEncoder, indent=2)

	@abstractmethod
	def save_plots(self):
		pass

	def _encode(self, inputs: list[str]):
		"""Encode a batch of strings, returning (input_ids, attention_mask) on device.

		Handles both HuggingFace tokenizers (with built-in left-padding) and bare
		spm.SentencePieceProcessor (which requires manual left-padding).
		"""
		if isinstance(self.tokenizer, spm.SentencePieceProcessor):
			encoded = [self.tokenizer.Encode(t) for t in inputs]
			max_len = max(len(e) for e in encoded)
			pad_id = 0
			input_ids = torch.tensor(
				[[pad_id] * (max_len - len(e)) + e for e in encoded], dtype=torch.long
			).to(self.device)
			attention_mask = torch.tensor(
				[[0] * (max_len - len(e)) + [1] * len(e) for e in encoded], dtype=torch.long
			).to(self.device)
			return input_ids, attention_mask
		else:
			tokens = self.tokenizer(inputs, return_tensors="pt", padding=True)
			return tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device)

	def _decode(self, token_ids: list[int]) -> str:
		"""Decode a list of token IDs to a string for either tokenizer type."""
		if isinstance(self.tokenizer, spm.SentencePieceProcessor):
			return self.tokenizer.Decode(token_ids)
		else:
			return self.tokenizer.batch_decode([token_ids], skip_special_tokens=True)[0]

	def _pad_token_id(self) -> int:
		"""Return a safe pad_token_id for model.generate()."""
		if isinstance(self.tokenizer, spm.SentencePieceProcessor):
			return 0
		else:
			return self.model.config.eos_token_id

	def _generate_single(self, input_ids, attention_mask, max_new_tokens: int):
		"""Call model.generate() with the right interface for each model family.

		MambaLMHeadModel (mamba_ssm) uses a custom generate API:
		- max_length (absolute) instead of max_new_tokens (relative)
		- cg=False (CUDA graphs are disabled for correctness / stability)
		- no attention_mask or pad_token_id (SSM has no attention mechanism)

		All other models use the standard HuggingFace generate API.
		"""
		if type(self.model).__name__ == "MambaLMHeadModel":
			return self.model.generate(
				input_ids=input_ids,
				max_length=input_ids.shape[1] + max_new_tokens,
				cg=False,
				eos_token_id=self.tokenizer.eos_token_id,
			)
		else:
			return self.model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				max_new_tokens=max_new_tokens,
				do_sample=False,
				pad_token_id=self._pad_token_id(),
			)

	def _generate(self, input_ids, attention_mask, max_new_tokens: int):
		"""Sub-batch generation to bound peak GPU memory.

		Large batches with long sequences can OOM due to the KV cache scaling with
		batch_size × seq_len × num_layers.  We split into sub-batches of at most
		TRANSFORMER_MAX_BATCH or MAMBA_MAX_BATCH (chosen from args.model), generate
		each, and concatenate.
		"""
		max_batch = TRANSFORMER_MAX_BATCH if self.args.model in TRANSFORMER_MODELS else MAMBA_MAX_BATCH
		batch_size = input_ids.shape[0]
		if batch_size <= max_batch:
			return self._generate_single(input_ids, attention_mask, max_new_tokens)

		outputs = []
		for start in range(0, batch_size, max_batch):
			end = min(start + max_batch, batch_size)
			sub_ids = input_ids[start:end]
			sub_mask = attention_mask[start:end] if attention_mask is not None else None
			out = self._generate_single(sub_ids, sub_mask, max_new_tokens)
			outputs.append(out)
			del sub_ids, sub_mask
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

		return torch.cat(outputs, dim=0)

	def _get_dataset(self, length, target_entry_index=None):
		if self.args.dataset in ("phonebook", "phonebook_local"):
			dataset_cls = PhonebookDataset
		elif self.args.dataset in ("profbook", "profbook_local"):
			dataset_cls = ProfbookDataset
		else:
			raise ValueError(f"Dataset {self.args.dataset} not supported")
		return dataset_cls(
			task=self.args.task,
			length=length,
			batch_size=self.args.eval_batch_size,
			set_size=self.args.eval_num_batches * self.args.eval_batch_size,
			target_entry_index=target_entry_index,
			args=self.args,
		)

	@property
	def _experiment_name(self):
		return re.sub(r'(?<!^)(?=[A-Z])', '_', type(self).__name__).lower()

	@property
	def _experiment_subdir(self):
		"""Relative path segment shared by both output roots: '{model}_{task}/{experiment_name}/{dataset}'."""
		return os.path.join(
			f"{self.args.model}_{self.args.task}",
			self._experiment_name,
			self.args.dataset,
		)

	def _get_plots_path(self):
		return experiment_plots_dir(self._experiment_subdir, *self._output_path_extra)

	def _get_raw_results_path(self):
		return experiment_raw_results_dir(self._experiment_subdir, *self._output_path_extra)
