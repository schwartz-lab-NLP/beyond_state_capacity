"""Gate Probing experiment.

Train per-layer linear probes on Mamba2 gate vectors (dt_raw or delta) at a
chosen focus token of each phonebook entry. Task: given the per-head gate
vector, predict whether that entry is the target.

Collection reuses SelectionDistinction's in_proj component and char-span token
mapping. Probe training and evaluation are inherited from RepresentationProbing.

Requirements
------------
Mamba models only. --task must be prefix or both.
--selectivity-component must be dt_raw or delta.

Configuration
-------------
  --gate-probing-entries-per-prompt  (default 150)
  --selectivity-component  dt_raw / delta
  --representation-probing-entry-part
      name_first_token / name_last_token / number_first_token /
      number_last_token (default).
  Hardcoded (from RepresentationProbing): NUM_SAMPLES=500, TRAIN_FRACTION=0.8,
  EPOCHS=100, LR=1e-3, BATCH_SIZE=1024, N_BOOTSTRAP=1000, CI_LEVEL=0.95.
  --skip-collection resumes from an existing data_index.json.

Protocol
--------
1. Collect: one random prompt per sample; hook in_proj outputs; write memmaps
   of the (n_heads,) gate vector at each entry's focus token plus labels
   and sample/entry ids. Train/test split is at the prompt level.
2. Train: one linear probe per layer with class-weighted BCE.
3. Evaluate vs the always-negative baseline; prompt-clustered bootstrap CIs.

Outputs
-------
results.json (per_layer metrics, split info, CI metadata),
{variant}test_predictions.npz, {variant}probe_weights.pt,
{variant}data_index.json, and per-metric PNG plots.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from libs.experiments.paper.representation_probing import (
	RepresentationProbing,
	_PromptEntryTokenization,
)
from libs.experiments.paper.selection_distinction import SelectionDistinction
from libs.models.models import MAMBA_MODELS


ALLOWED_COMPONENTS = ("dt_raw", "delta")


class GateProbing(RepresentationProbing, SelectionDistinction):
	"""Per-layer linear probes on dt_raw / delta at a chosen entry token."""

	def __init__(self, args, model, tokenizer):
		if args.model not in MAMBA_MODELS:
			raise ValueError(
				f"gate_probing is only supported for mamba models: {MAMBA_MODELS}"
			)
		if args.selectivity_component not in ALLOWED_COMPONENTS:
			raise ValueError(
				f"gate_probing only supports --selectivity-component "
				f"{' or '.join(ALLOWED_COMPONENTS)} "
				f"(got {args.selectivity_component!r})"
			)
		super().__init__(args, model, tokenizer)

	@property
	def _variant_prefix(self):
		prefix = (
			f"{self.args.selectivity_component}_"
			f"{self.args.representation_probing_entry_part}_"
		)
		if self.args.dataset.endswith("_local"):
			prefix += f"{self.args.local_dataset_flag_entries}_"
		return prefix

	@property
	def _entries_per_prompt(self):
		return self.args.gate_probing_entries_per_prompt

	@property
	def _metrics_plot_title_suffix(self):
		return (
			f"({self.args.model}, {self.args.task}, "
			f"{self.args.selectivity_component}, "
			f"{self.args.representation_probing_entry_part}, "
			f"entries={self._entries_per_prompt})"
		)

	def run(self):
		super().run()
		self.results["component"] = self.args.selectivity_component
		self.results["entry_part"] = self.args.representation_probing_entry_part

	def _collect_data(self, raw_path: str) -> Dict:
		"""Collect per-head gate vectors to disk using per-layer memmaps."""
		n_layers = int(self.model.config.n_layer)
		n_heads = int(self.model.backbone.layers[0].mixer.nheads)
		total_entries = self.NUM_SAMPLES * self._entries_per_prompt

		rep_dir = os.path.join(raw_path, f"{self._variant_prefix}gates")
		Path(rep_dir).mkdir(parents=True, exist_ok=True)

		labels_path = os.path.join(rep_dir, "labels.npy")
		sample_id_path = os.path.join(rep_dir, "sample_ids.npy")
		entry_id_path = os.path.join(rep_dir, "entry_ids.npy")

		labels_mm = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.int8, shape=(total_entries,))
		sample_ids_mm = np.lib.format.open_memmap(sample_id_path, mode="w+", dtype=np.int32, shape=(total_entries,))
		entry_ids_mm = np.lib.format.open_memmap(entry_id_path, mode="w+", dtype=np.int16, shape=(total_entries,))

		layer_paths = {}
		layer_mms = {}
		for l in range(n_layers):
			p = os.path.join(rep_dir, f"layer_{l:03d}.npy")
			layer_paths[str(l)] = p
			layer_mms[l] = np.lib.format.open_memmap(
				p, mode="w+", dtype=np.float16, shape=(total_entries, n_heads)
			)

		write_offset = 0

		self.args.eval_batch_size = 1
		dataset = self._get_dataset(self._entries_per_prompt)

		hook_storage: Dict[int, torch.Tensor] = {}
		hooks = self._register_gate_hooks(n_layers, hook_storage)
		try:
			for sample_idx in tqdm(range(self.NUM_SAMPLES), desc="Collect gates"):
				batch = dataset[0]
				prompt = batch["input"][0]
				prompt_parts = batch["prompt_parts"][0]
				target_index = int(batch["target_index"][0])

				input_ids, attention_mask = self._encode([prompt])
				ids = input_ids[0, attention_mask[0].bool()].tolist()
				pad_offset = int(input_ids.shape[1] - attention_mask[0].sum().item())
				tok = self._focus_tokens_from_ids(
					prompt_parts, target_index, ids, pad_offset,
				)
				positions = tok.entry_focus_token_positions
				entry_labels = tok.entry_is_target_labels
				assert len(positions) == self._entries_per_prompt
				assert len(entry_labels) == self._entries_per_prompt

				hook_storage.clear()
				with torch.inference_mode():
					_ = self.model(input_ids=input_ids)

				for i in range(self._entries_per_prompt):
					labels_mm[write_offset + i] = np.int8(entry_labels[i])
					sample_ids_mm[write_offset + i] = np.int32(sample_idx)
					entry_ids_mm[write_offset + i] = np.int16(i)

				for l in range(n_layers):
					gates_full = hook_storage[l]  # (1, T, n_heads) on CPU float16
					gates_sel = gates_full[0, positions, :]  # (entries, n_heads)
					layer_mms[l][write_offset : write_offset + self._entries_per_prompt] = gates_sel.numpy()

				write_offset += self._entries_per_prompt

		finally:
			for h in hooks:
				h.remove()

		del labels_mm, sample_ids_mm, entry_ids_mm
		for l in range(n_layers):
			del layer_mms[l]

		index = {
			"rep_dir": rep_dir,
			"total_entries": total_entries,
			"n_layers": n_layers,
			"n_heads": n_heads,
			"component": self.args.selectivity_component,
			"entry_part": self.args.representation_probing_entry_part,
			"layer_paths": layer_paths,
			"labels_path": labels_path,
			"sample_ids_path": sample_id_path,
			"entry_ids_path": entry_id_path,
		}
		with open(os.path.join(raw_path, f"{self._variant_prefix}data_index.json"), "w") as f:
			json.dump(index, f, indent=2)

		print(f"Saved collected gates under: {rep_dir}")
		return index

	def _register_gate_hooks(self, n_layers: int, hook_storage: Dict[int, torch.Tensor]):
		"""Capture dt_raw / delta from mixer.in_proj for each layer.

		Hook stores a CPU float16 tensor of shape (B, T, n_heads) for each layer.
		"""
		hooks = []
		for l in range(n_layers):
			mixer = self.model.backbone.layers[l].mixer
			dt_bias = mixer.dt_bias.detach().clone()
			A = -torch.exp(mixer.A_log.float()).detach().clone()
			n_heads = mixer.nheads
			d_ssm = mixer.d_ssm
			ngroups = mixer.ngroups
			d_state = mixer.d_state

			def make_hook(layer_idx, dt_bias, A, n_heads, d_ssm, ngroups, d_state):
				def hook_fn(module, input, output):
					component = self._compute_component(
						output.float(),
						dt_bias=dt_bias.float(), A=A, n_heads=n_heads,
						d_ssm=d_ssm, ngroups=ngroups, d_state=d_state,
					)
					if component.dim() == 2:
						component = component.unsqueeze(1)
					hook_storage[layer_idx] = component.detach().to("cpu", dtype=torch.float16)
				return hook_fn

			hooks.append(mixer.in_proj.register_forward_hook(
				make_hook(l, dt_bias, A, n_heads, d_ssm, ngroups, d_state)
			))
		return hooks

	def _focus_tokens_from_ids(
		self,
		prompt_parts,
		target_entry_index: int,
		ids: List[int],
		pad_offset: int,
	) -> _PromptEntryTokenization:
		"""One focus token per entry via SelectionDistinction char-span ranges."""
		out = _PromptEntryTokenization()
		entry_part = self.args.representation_probing_entry_part
		use_name = entry_part in ("name_first_token", "name_last_token")
		use_first = entry_part in ("name_first_token", "number_first_token")

		for is_target, name_range, val_range in self._iter_entry_token_ranges(
			prompt_parts, target_entry_index, ids,
		):
			token_range = name_range if use_name else val_range
			if not token_range:
				raise RuntimeError(
					f"Empty {'name' if use_name else 'number'} token range "
					f"for entry_part={entry_part!r}"
				)
			pos = token_range[0] if use_first else token_range[-1]
			out.entry_focus_token_positions.append(pos + pad_offset)
			out.entry_is_target_labels.append(int(is_target))
		return out
