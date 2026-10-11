"""Representation Probing experiment.

Train per-layer linear probes on entry-token representations (the input to
mixer.in_proj). Task: given the representation at a chosen focus token of each
entry, predict whether that entry is the target.

Requirements
------------
Mamba models only. --task must be prefix or both.

Configuration
-------------
  --representation-probing-entries-per-prompt  (default 150)
  --representation-probing-entry-part
      name_first_token / name_last_token / number_first_token /
      number_last_token (default).
  Hardcoded: NUM_SAMPLES=500, TRAIN_FRACTION=0.8, EPOCHS=100, LR=1e-3,
  BATCH_SIZE=1024, N_BOOTSTRAP=1000, CI_LEVEL=0.95.
  --skip-collection resumes from an existing data_index.json.

Protocol
--------
1. Collect: one random prompt per sample; hook in_proj inputs; write
   memmaps of representations at each entry's focus token plus labels
   and sample/entry ids. Train/test split is at the prompt level.
2. Train: one linear probe per layer with class-weighted BCE.
3. Evaluate vs the always-negative baseline; prompt-clustered bootstrap CIs.

Metric
------
Per-layer acc / precision / recall / F1 / AUC, plus rescue_rate
(P(probe=1|y=1)) and damage_rate (P(probe=1|y=0)).

Outputs
-------
results.json (per_layer metrics, split info, CI metadata),
{variant}test_predictions.npz, {variant}probe_weights.pt,
{variant}data_index.json, and per-metric PNG plots.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from libs.experiments.experiment import Experiment
from libs.models.models import MAMBA_MODELS


@dataclass
class _PromptEntryTokenization:
	entry_focus_token_positions: List[int] = field(default_factory=list)
	entry_is_target_labels: List[int] = field(default_factory=list)


class RepresentationProbing(Experiment):
	"""Per-layer linear probes on mixer.in_proj inputs at a chosen entry token."""

	NUM_SAMPLES = 500
	TRAIN_FRACTION = 0.8
	EPOCHS = 100
	LR = 1e-3
	BATCH_SIZE = 1024
	N_BOOTSTRAP = 1000
	CI_LEVEL = 0.95

	@property
	def _variant_prefix(self):
		if self.args.dataset.endswith("_local"):
			return f"{self.args.local_dataset_flag_entries}_"
		return ""

	@property
	def _entries_per_prompt(self):
		return self.args.representation_probing_entries_per_prompt

	@property
	def _metrics_plot_title_suffix(self):
		return (
			f"({self.args.model}, {self.args.task}, "
			f"entries={self._entries_per_prompt})"
		)

	def run(self):
		if self.args.model not in MAMBA_MODELS:
			raise ValueError(f"{self._experiment_name} is only supported for mamba models: {MAMBA_MODELS}")
		if self.args.task not in ["prefix", "both"]:
			raise ValueError(f"{self._experiment_name} is defined for task=prefix or task=both (phonebook prefix/both variations)")

		banner = self._experiment_name.replace("_", " ").upper()
		print(f"{'#' * 80}")
		print(f"{'#' * 16} {banner} — SAMPLES: {self.NUM_SAMPLES} {'#' * 16}")
		print(f"{'#' * 80}")

		self.model.eval()
		raw_path = self._get_raw_results_path()
		plots_path = self._get_plots_path()

		data_index_path = os.path.join(raw_path, f"{self._variant_prefix}data_index.json")
		if self.args.skip_collection:
			if not os.path.exists(data_index_path):
				raise FileNotFoundError(
					f"--skip-collection was set, but {data_index_path} does not exist."
				)
			with open(data_index_path, "r") as f:
				data_index = json.load(f)
			print(f"Skipping data collection; loaded existing data index: {data_index_path}")
		else:
			data_index = self._collect_data(raw_path)

		results, split_info = self._train_and_evaluate_probes(raw_path, data_index)
		baseline_acc = split_info["baseline_acc"]

		self.results = {
			"n_samples": self.NUM_SAMPLES,
			"entries_per_prompt": self._entries_per_prompt,
			"total_entries": int(data_index["total_entries"]),
			"train_fraction": self.TRAIN_FRACTION,
			"epochs": self.EPOCHS,
			"lr": self.LR,
			"batch_size": self.BATCH_SIZE,
			"split": {
				"unit": "prompt",
				"n_train_prompts": split_info["n_train_prompts"],
				"n_test_prompts": split_info["n_test_prompts"],
				"n_train_entries": split_info["n_train_entries"],
				"n_test_entries": split_info["n_test_entries"],
			},
			"trivial_baseline": {
				"acc": float(baseline_acc),
				"f1": 0.0,
				"auc": 0.5,
				"rescue_rate": 0.0,
				"damage_rate": 0.0,
				"note": (
					"always-predict-negative classifier. Rescue rate is recall among "
					"true targets (baseline failures); damage rate is the false-positive "
					"rate among non-targets (baseline successes)."
				),
			},
			"predictions_path": split_info["predictions_path"],
			"ci": {
				"method": "prompt-clustered percentile bootstrap",
				"n_bootstrap": self.N_BOOTSTRAP,
				"level": self.CI_LEVEL,
			},
			"per_layer": results,
		}

		self._save_metrics_plots(results, plots_path)
		print(f"Saved metric plots to: {plots_path}")

	def save_plots(self):
		# Plots are saved during run() since they depend on per-layer probe metrics.
		return

	def _collect_data(self, raw_path: str) -> Dict:
		"""Collect representations to disk using per-layer memmaps."""
		n_layers = int(self.model.config.n_layer)
		d_model = int(self.model.backbone.layers[0].mixer.in_proj.in_features)
		total_entries = self.NUM_SAMPLES * self._entries_per_prompt

		rep_dir = os.path.join(raw_path, f"{self._variant_prefix}representations")
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
				p, mode="w+", dtype=np.float16, shape=(total_entries, d_model)
			)

		write_offset = 0

		self.args.eval_batch_size = 1
		dataset = self._get_dataset(self._entries_per_prompt)

		hook_storage: Dict[int, torch.Tensor] = {}
		hooks = self._register_u_hooks(n_layers, hook_storage)
		try:
			for sample_idx in tqdm(range(self.NUM_SAMPLES), desc="Collect representations"):
				# dataset generates a fresh random batch on each call; we take index [0]
				# to process exactly one sample per forward pass regardless of batch size.
				batch = dataset[0]
				prompt = batch["input"][0]
				prompt_parts = batch["prompt_parts"][0]
				target_index = int(batch["target_index"][0])
				tok = self._tokenize_prompt_entries(prompt_parts, target_index)
				positions = tok.entry_focus_token_positions
				entry_labels = tok.entry_is_target_labels
				assert len(positions) == self._entries_per_prompt
				assert len(entry_labels) == self._entries_per_prompt

				# Forward pass (full sequence) and capture per-layer in_proj inputs.
				input_ids, _ = self._encode([prompt])
				hook_storage.clear()
				with torch.inference_mode():
					_ = self.model(input_ids=input_ids)

				# Write to memmaps.
				for i in range(self._entries_per_prompt):
					labels_mm[write_offset + i] = np.int8(entry_labels[i])
					sample_ids_mm[write_offset + i] = np.int32(sample_idx)
					entry_ids_mm[write_offset + i] = np.int16(i)

				for l in range(n_layers):
					u_full = hook_storage[l]  # (1, T, d_model) on CPU float16
					u_sel = u_full[0, positions, :]  # (entries, d_model)
					layer_mms[l][write_offset : write_offset + self._entries_per_prompt] = u_sel.numpy()

				write_offset += self._entries_per_prompt

		finally:
			for h in hooks:
				h.remove()

		# Flush memmaps
		del labels_mm, sample_ids_mm, entry_ids_mm
		for l in range(n_layers):
			del layer_mms[l]

		index = {
			"rep_dir": rep_dir,
			"total_entries": total_entries,
			"n_layers": n_layers,
			"d_model": d_model,
			"layer_paths": layer_paths,
			"labels_path": labels_path,
			"sample_ids_path": sample_id_path,
			"entry_ids_path": entry_id_path,
		}
		with open(os.path.join(raw_path, f"{self._variant_prefix}data_index.json"), "w") as f:
			json.dump(index, f, indent=2)

		print(f"Saved collected representations under: {rep_dir}")
		return index

	def _register_u_hooks(self, n_layers: int, hook_storage: Dict[int, torch.Tensor]):
		"""Capture u = input to mixer.in_proj for each layer.

		Hook stores a CPU float16 tensor of shape (B, T, d_model) for each layer.
		"""
		hooks = []
		for l in range(n_layers):
			mixer = self.model.backbone.layers[l].mixer

			def make_hook(layer_idx: int):
				def hook_fn(module, input, output):
					# input[0] is the tensor flowing into in_proj (u).
					u = input[0]
					if u.dim() == 2:
						u = u.unsqueeze(1)
					hook_storage[layer_idx] = u.detach().to("cpu", dtype=torch.float16)
				return hook_fn

			hooks.append(mixer.in_proj.register_forward_hook(make_hook(l)))
		return hooks

	def _tokenize_prompt_entries(self, prompt_parts, target_entry_index: int) -> _PromptEntryTokenization:
		"""Locate one focus token per entry and a binary is-target label.

		Which token is used is --representation-probing-entry-part:
		name_first_token, name_last_token, number_first_token, or
		number_last_token. Token counts come from tokenizing the growing
		prompt prefix (HuggingFace .input_ids or a SentencePiece id list).
		"""
		out = _PromptEntryTokenization()

		entry_part = self.args.representation_probing_entry_part
		name_first_token = entry_part == "name_first_token"
		name_last_token = entry_part == "name_last_token"
		number_first_token = entry_part == "number_first_token"
		number_last_token = entry_part == "number_last_token"

		prefix = prompt_parts["prefix"]
		entries = prompt_parts["entries"]
		suffix = prompt_parts["suffix"]

		prompt = ""
		num_tokens = 0

		def token_count(text: str) -> int:
			# Mirror other experiments: `self.tokenizer(prompt).input_ids` works for both HF and spm.
			enc = self.tokenizer(text)
			return len(enc.input_ids) if hasattr(enc, "input_ids") else len(enc)

		def append(text: str) -> None:
			nonlocal prompt, num_tokens
			prompt += text
			num_tokens = token_count(prompt)

		append(prefix)

		for entry_idx, entry in enumerate(entries):
			entry_name, entry_number = entry.rstrip().split(": ", 1)
			is_target = int(entry_idx == target_entry_index)

			if name_first_token:
				focus_pos = num_tokens
			append(entry_name)
			if name_last_token:
				focus_pos = num_tokens - 1

			append(": ")

			if number_first_token:
				focus_pos = num_tokens
			append(entry_number)
			if number_last_token:
				focus_pos = num_tokens - 1

			out.entry_focus_token_positions.append(focus_pos)
			out.entry_is_target_labels.append(is_target)

			append("\n")

		append(suffix)
		return out

	def _train_and_evaluate_probes(self, raw_path: str, data_index: Dict):
		n_layers = int(data_index["n_layers"])
		total_entries = int(data_index["total_entries"])

		y = np.load(data_index["labels_path"], mmap_mode="r").astype(np.int64)
		if "sample_ids_path" not in data_index:
			raise KeyError(
				"data_index is missing sample_ids_path; re-run collection. "
				"Prompt-level splits require per-entry sample ids."
			)
		sample_ids = np.load(data_index["sample_ids_path"], mmap_mode="r")
		entry_ids = np.load(data_index["entry_ids_path"], mmap_mode="r")

		rng = np.random.default_rng(int(self.args.seed))
		train_idx, test_idx, n_train_prompts, n_test_prompts = self._prompt_level_split(
			sample_ids, rng
		)

		n_pos_all = int((y == 1).sum())
		n_neg_all = int((y == 0).sum())
		n_pos_train = int((y[train_idx] == 1).sum())
		n_neg_train = int((y[train_idx] == 0).sum())
		pos_weight = float(n_neg_train / max(n_pos_train, 1))
		pos_weight_t = torch.tensor([pos_weight], device=self.device, dtype=torch.float32)

		test_y = np.asarray(y[test_idx], dtype=np.int64)
		test_sample_ids = np.asarray(sample_ids[test_idx], dtype=np.int32)
		test_entry_ids = np.asarray(entry_ids[test_idx], dtype=np.int16)
		n_test_pos = int((test_y == 1).sum())
		n_test_neg = int((test_y == 0).sum())
		baseline_acc = n_test_neg / max(len(test_idx), 1)

		print(f"Total entries: {total_entries}  (prompt-level split)")
		print(f"Prompts: train={n_train_prompts}  test={n_test_prompts}")
		print(f"Positives: {n_pos_all}  Negatives: {n_neg_all}  pos_weight={pos_weight:.3f}")
		print(f"Train: {len(train_idx)}  Test: {len(test_idx)}  (test pos={n_test_pos}, neg={n_test_neg})")
		print(f"Trivial baseline (always-no): acc={baseline_acc:.4f}  f1=0.0000  auc=0.5000")

		prompt_groups = self._group_indices_by_id(test_sample_ids)
		boot_rng = np.random.default_rng(int(self.args.seed) + 1)
		boot_idx_list = self._bootstrap_prompt_indices(prompt_groups, boot_rng)

		weights_out = {}
		per_layer_results = {}
		test_logits = np.zeros((len(test_idx), n_layers), dtype=np.float32)
		test_pred = np.zeros((len(test_idx), n_layers), dtype=np.int8)

		for l in range(n_layers):
			x_path = data_index["layer_paths"][str(l)]
			x_mm = np.load(x_path, mmap_mode="r")  # (N, d_model) float16
			d_model = x_mm.shape[1]

			probe = nn.Linear(d_model, 1, bias=True).to(self.device, dtype=torch.float32)
			opt = torch.optim.Adam(probe.parameters(), lr=self.LR)
			criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

			epoch_losses = []
			probe.train()
			for epoch in range(self.EPOCHS):
				rng.shuffle(train_idx)
				batches = 0
				loss_sum = 0.0

				for start in range(0, len(train_idx), self.BATCH_SIZE):
					idx = train_idx[start : start + self.BATCH_SIZE]
					xb = torch.from_numpy(np.asarray(x_mm[idx], dtype=np.float32)).to(self.device)
					yb = torch.from_numpy(y[idx].astype(np.float32)).to(self.device).unsqueeze(1)

					logits = probe(xb)
					loss = criterion(logits, yb)
					opt.zero_grad(set_to_none=True)
					loss.backward()
					opt.step()

					loss_sum += float(loss.item())
					batches += 1

				epoch_losses.append(loss_sum / max(batches, 1))

			probe.eval()
			with torch.inference_mode():
				layer_scores = []
				for start in range(0, len(test_idx), self.BATCH_SIZE):
					idx = test_idx[start : start + self.BATCH_SIZE]
					xb = torch.from_numpy(np.asarray(x_mm[idx], dtype=np.float32)).to(self.device)
					layer_scores.append(probe(xb).squeeze(1).detach().cpu().numpy())
				layer_scores = np.concatenate(layer_scores, axis=0).astype(np.float32)

			prob = 1.0 / (1.0 + np.exp(-layer_scores))
			pred = (prob >= 0.5).astype(np.int64)
			test_logits[:, l] = layer_scores
			test_pred[:, l] = pred.astype(np.int8)

			stats = self._classification_stats(test_y, pred)
			try:
				auc = float(roc_auc_score(test_y, prob))
			except ValueError:
				auc = float("nan")
			ci = self._bootstrap_metric_cis(test_y, pred, boot_idx_list)

			cm = stats["confusion"]
			print(
				f"Layer {l:03d} | "
				f"acc={stats['acc']:.4f} prec={stats['precision']:.4f} rec={stats['recall']:.4f} "
				f"f1={stats['f1']:.4f} auc={auc:.4f} | "
				f"rescue={stats['rescue_rate']:.4f} damage={stats['damage_rate']:.4f} "
				f"acc-base={stats['acc_minus_baseline']:.4f} | "
				f"tp={cm['tp']} fp={cm['fp']} fn={cm['fn']} tn={cm['tn']}"
			)

			per_layer_results[str(l)] = {
				"acc": stats["acc"],
				"precision": stats["precision"],
				"recall": stats["recall"],
				"f1": stats["f1"],
				"auc": auc,
				"rescue_rate": stats["rescue_rate"],
				"damage_rate": stats["damage_rate"],
				"acc_minus_baseline": stats["acc_minus_baseline"],
				"confusion": cm,
				"ci": ci,
			}

			weights_out[str(l)] = {
				"state_dict": {k: v.detach().cpu() for k, v in probe.state_dict().items()},
			}

		predictions_path = os.path.join(raw_path, f"{self._variant_prefix}test_predictions.npz")
		np.savez_compressed(
			predictions_path,
			sample_id=test_sample_ids,
			entry_id=test_entry_ids,
			y=test_y.astype(np.int8),
			logits=test_logits,
			pred=test_pred,
			train_idx=train_idx.astype(np.int32),
			test_idx=test_idx.astype(np.int32),
		)
		print(f"Saved per-example test predictions to: {predictions_path}")

		weights_path = os.path.join(raw_path, f"{self._variant_prefix}probe_weights.pt")
		torch.save(
			{
				"experiment": self._experiment_name,
				"n_layers": n_layers,
				"weights": weights_out,
			},
			weights_path,
		)
		print(f"Saved probe weights to: {weights_path}")

		split_info = {
			"baseline_acc": float(baseline_acc),
			"n_train_prompts": int(n_train_prompts),
			"n_test_prompts": int(n_test_prompts),
			"n_train_entries": int(len(train_idx)),
			"n_test_entries": int(len(test_idx)),
			"predictions_path": predictions_path,
		}
		return per_layer_results, split_info

	def _prompt_level_split(
		self, sample_ids: np.ndarray, rng: np.random.Generator
	) -> Tuple[np.ndarray, np.ndarray, int, int]:
		"""Assign whole prompts to train or test. Entries from one prompt never cross the split."""
		prompts = np.unique(np.asarray(sample_ids))
		rng.shuffle(prompts)
		n_train_prompts = int(len(prompts) * self.TRAIN_FRACTION)
		train_mask = np.isin(np.asarray(sample_ids), prompts[:n_train_prompts])
		train_idx = np.flatnonzero(train_mask)
		test_idx = np.flatnonzero(~train_mask)
		train_prompts = np.unique(np.asarray(sample_ids)[train_idx])
		test_prompts = np.unique(np.asarray(sample_ids)[test_idx])
		if np.intersect1d(train_prompts, test_prompts).size:
			raise RuntimeError("Prompt-level split leaked prompts across train and test.")
		rng.shuffle(train_idx)
		rng.shuffle(test_idx)
		return train_idx, test_idx, n_train_prompts, int(len(prompts) - n_train_prompts)

	@staticmethod
	def _group_indices_by_id(ids: np.ndarray) -> List[np.ndarray]:
		order = np.argsort(ids, kind="mergesort")
		sorted_ids = ids[order]
		cuts = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
		return np.split(order, cuts)

	def _bootstrap_prompt_indices(
		self, prompt_groups: Sequence[np.ndarray], rng: np.random.Generator
	) -> List[np.ndarray]:
		n_prompts = len(prompt_groups)
		draws = rng.integers(0, n_prompts, size=(self.N_BOOTSTRAP, n_prompts))
		return [np.concatenate([prompt_groups[i] for i in draw]) for draw in draws]

	@staticmethod
	def _classification_stats(y: np.ndarray, pred: np.ndarray) -> Dict:
		"""Entry-level binary stats vs the always-negative baseline.

		Rescue rate = P(probe=1 | y=1): fraction of baseline failures the probe recovers.
		Damage rate = P(probe=1 | y=0): fraction of baseline successes the probe breaks.
		acc_minus_baseline is the paired accuracy gap on the same examples.
		"""
		tp = int(((pred == 1) & (y == 1)).sum())
		tn = int(((pred == 0) & (y == 0)).sum())
		fp = int(((pred == 1) & (y == 0)).sum())
		fn = int(((pred == 0) & (y == 1)).sum())
		n = max(len(y), 1)
		n_pos = tp + fn
		n_neg = tn + fp
		acc = (tp + tn) / n
		baseline_acc = n_neg / n
		precision = tp / max(tp + fp, 1)
		recall = tp / max(n_pos, 1)
		f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
		return {
			"acc": float(acc),
			"precision": float(precision),
			"recall": float(recall),
			"f1": float(f1),
			"rescue_rate": float(recall),
			"damage_rate": float(fp / max(n_neg, 1)),
			"acc_minus_baseline": float(acc - baseline_acc),
			"confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
		}

	def _bootstrap_metric_cis(
		self,
		y: np.ndarray,
		pred: np.ndarray,
		boot_idx_list: Sequence[np.ndarray],
	) -> Dict[str, Dict[str, float]]:
		"""Percentile CIs from prompt-clustered bootstrap (paired with the always-negative baseline)."""
		metric_names = ("f1", "rescue_rate", "damage_rate", "acc_minus_baseline")
		collected = {name: np.empty(len(boot_idx_list), dtype=np.float64) for name in metric_names}
		for b, idx in enumerate(boot_idx_list):
			stats = self._classification_stats(y[idx], pred[idx])
			for name in metric_names:
				collected[name][b] = stats[name]
		alpha = (1.0 - self.CI_LEVEL) / 2.0
		cis = {}
		for name in metric_names:
			lo, hi = np.quantile(collected[name], [alpha, 1.0 - alpha])
			cis[name] = {
				"low": float(lo),
				"high": float(hi),
				"level": float(self.CI_LEVEL),
			}
		return cis

	def _save_metrics_plots(self, per_layer_results: Dict[str, Dict], plots_path: str):
		n_layers = len(per_layer_results)
		layers = np.arange(n_layers)
		title_suffix = self._metrics_plot_title_suffix

		for metric_name, ylabel, ylim in [
			("f1", "F1 score", (0, 1)),
			("precision", "Precision", (0, 1)),
			("recall", "Recall", (0, 1)),
			("rescue_rate", "Rescue rate (baseline failures)", (0, 1)),
			("damage_rate", "Damage rate (baseline successes)", (0, 1)),
			("acc_minus_baseline", "Accuracy minus trivial baseline", None),
		]:
			values = np.array([per_layer_results[str(l)][metric_name] for l in range(n_layers)])

			fig, ax = plt.subplots(figsize=(10, 6))
			ax.plot(layers, values, marker="o", linewidth=2)
			ci0 = per_layer_results["0"].get("ci", {})
			if metric_name in ci0:
				lo = np.array([per_layer_results[str(l)]["ci"][metric_name]["low"] for l in range(n_layers)])
				hi = np.array([per_layer_results[str(l)]["ci"][metric_name]["high"] for l in range(n_layers)])
				ax.fill_between(layers, lo, hi, alpha=0.25)
			ax.set_xlabel("Layer")
			ax.set_ylabel(ylabel)
			ax.set_title(f"Probe {ylabel} vs Layer (test set)\n{title_suffix}")
			ax.set_xticks(layers)
			if ylim is not None:
				ax.set_ylim(*ylim)
			ax.grid(True, alpha=0.3)

			out_path = os.path.join(plots_path, f"{self._variant_prefix}{metric_name}.png")
			plt.tight_layout()
			plt.savefig(out_path, dpi=150)
			plt.close()
			print(f"Saved {metric_name} plot to: {out_path}")

