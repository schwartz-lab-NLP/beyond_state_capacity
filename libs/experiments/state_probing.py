"""State Probing experiment.

Per-layer linear probes: does the SSM recurrent state encode the target
entry's profession?

Online fused mode: Mamba forward passes and probe training are interleaved
in a single streaming pass — states are never written to disk.

Phase 1 (train, first TRAIN_FRACTION of prompts, batches of MAMBA_FWD_BATCH
  prompts stepped together): capture per-layer SSM states, feed each into
  a class-balanced replay reservoir from which the SGD ProbeBank is trained
  with balanced minibatches on standardized features.

Phase 2 (eval, remaining prompts): same forward/capture; each state is
  scored by the SGD probe.

Phase 3 (plots): accuracy vs layer, accuracy vs profession, heatmap over
  target_entry_index x post_target_entries.

Lessons from previous runs baked into this design:
  - A FIFO replay buffer over ~4 prompts collapses training (loss pins at
    the buffer's label entropy, probes predict one class) -> balanced
    per-class reservoir + balanced minibatches.
  - Raw states are badly scaled -> standardize with global feature mean/std
    (frozen after NORM_WARMUP_STATES states).
  - Adam L2 with no signal actively drives W -> 0 -> weight decay 0 in raw
    mode.
  - B=1 token-by-token forwards dominate runtime -> batch MAMBA_FWD_BATCH
    prompts per forward pass (Mamba is causal; right-padding is harmless
    because every capture happens before a prompt's padding starts).

JL projection (--state-projection jl) reduces the ~1M-dim SSM state to
PROJECTION_DIM dims before probing; raw (--state-projection none) probes
the full state.

--state-probing-num-gpus:
  1: everything on one GPU (JL mode only; raw mode does not fit).
  2: Mamba on cuda:0; probes and staging on cuda:1;
     raw-mode reservoir in CPU RAM (see _BalancedReservoir).

Requirements
------------
Mamba models only. --dataset must be profbook or profbook_local.
Raw (unprojected) probing requires --state-probing-num-gpus 2.

Configuration
-------------
  --state-probing-entries-per-prompt  (default 150)
  --state-projection {none,jl} / --state-projection-dim (default 4096)
  --state-probing-num-gpus {1,2}
  --state-probing-reservoir-slots (optional override)
  --skip-collection / --run-sha to resume eval from train_checkpoint.pt
  Hardcoded: NUM_SAMPLES=15000, TRAIN_FRACTION=0.8, LR=1e-3,
  NUM_PROFESSIONS=20, MAMBA_FWD_BATCH=12, etc. (see class constants).

Metric
------
Per-layer probe accuracy (argmax profession vs ground-truth label) on
held-out prompts. Capture points are the last profession token of every
entry at or after the target.

Outputs
-------
results.json, probe_results.parquet, train_checkpoint.pt, and plots
under a run-SHA subdirectory.
"""

import gc
import os
import uuid
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from mamba_ssm.utils.generation import InferenceParams
from tqdm import tqdm

from libs.experiments.experiment import Experiment
from libs.models.models import MAMBA_MODELS


@dataclass
class _CapturePoint:
	token_position: int   # 0-indexed position of the last profession token
	entry_index: int      # absolute index of this entry in the phonebook
	post_target: int      # entry_index - target_entry_index


class _ProbeBank(nn.Module):
	"""Batched linear probes for all SSM layers in a single module.

	Holds W (n_layers, state_dim, num_classes) and b (n_layers, num_classes)
	as fp32 parameters.  A single Adam optimiser over this module trains all
	layer-probes simultaneously, avoiding 56 separate kernel launches per step.

	forward(x) where x is (B, n_layers, state_dim) fp32
	  → logits (B, n_layers, num_classes) fp32
	"""

	def __init__(self, n_layers: int, state_dim: int, num_classes: int):
		super().__init__()
		self.W = nn.Parameter(torch.empty(n_layers, state_dim, num_classes))
		self.b = nn.Parameter(torch.zeros(n_layers, num_classes))
		# kaiming_uniform init scaled to fan_in = state_dim (matching nn.Linear)
		bound = (3.0 ** 0.5) * (2.0 / 6.0) ** 0.5 / (state_dim ** 0.5)
		nn.init.uniform_(self.W, -bound, bound)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, n_layers, state_dim)
		x_T = x.permute(1, 0, 2)                    # (n_layers, B, state_dim)
		logits_T = torch.bmm(x_T, self.W)            # (n_layers, B, num_classes)
		return logits_T.permute(1, 0, 2) + self.b   # (B, n_layers, num_classes)


class _BalancedReservoir:
	"""Class-balanced reservoir of captured states.

	Motivation: a FIFO ring buffer over streamed prompts holds states from only
	the last few prompts, and since all captures of a prompt share one label,
	minibatches end up dominated by a handful of drifting labels.  The probes'
	optimal response to that data is to ignore the input and fit the label
	marginals with the bias — i.e. total collapse (exactly what we observed:
	train loss pinned at ln(#labels-in-buffer) and constant-class predictions).

	This reservoir instead keeps `slots_per_class` states per profession, each
	class maintained as an independent uniform reservoir sample over its stream
	(classic Algorithm-R).  Balanced minibatches drawn from it contain all
	classes, so the bias gains nothing and only genuine input signal can reduce
	the loss.

	Storage is fp16 with shape (num_classes * slots_per_class, n_layers, k).
	In raw mode it lives in CPU RAM (~2.35 GB per slot at 1M-dim x 56 layers
	fp16); in JL mode it lives on the probe GPU.
	"""

	def __init__(self, num_classes: int, slots_per_class: int, n_layers: int,
	             state_dim: int, device: str):
		self.num_classes = num_classes
		self.slots = slots_per_class
		shape = (num_classes * slots_per_class, n_layers, state_dim)
		nbytes = int(np.prod(shape)) * 2
		print(
			f"Allocating balanced reservoir: {num_classes} classes x {slots_per_class} slots "
			f"on {device} ({nbytes / 1e9:.1f} GB fp16) ...",
			flush=True,
		)
		if device == "cpu":
			# Skip pinning for large reservoirs: locked pages count toward the
			# cgroup memory limit and can trigger OOM on top of the allocation.
			use_pin = nbytes < 40e9
			if use_pin:
				try:
					self.X = torch.empty(shape, dtype=torch.float16, pin_memory=True)
				except RuntimeError as e:
					print(f"WARNING: could not pin reservoir memory ({e}); using pageable RAM.",
					      flush=True)
					self.X = torch.empty(shape, dtype=torch.float16)
			else:
				print("Reservoir >40 GB: using pageable CPU RAM (no pin_memory).",
				      flush=True)
				self.X = torch.empty(shape, dtype=torch.float16)
		else:
			self.X = torch.empty(shape, dtype=torch.float16, device=device)
		self.filled = np.zeros(num_classes, dtype=np.int64)
		self.seen   = np.zeros(num_classes, dtype=np.int64)

	@property
	def total_filled(self) -> int:
		return int(self.filled.sum())

	def insert(self, x16: torch.Tensor, c: int, rng: np.random.Generator) -> None:
		"""Insert one fp16 state (n_layers, k) for class c (reservoir sampling)."""
		if self.filled[c] < self.slots:
			pos = int(self.filled[c])
			self.filled[c] += 1
		else:
			j = int(rng.integers(0, self.seen[c] + 1))
			if j >= self.slots:
				self.seen[c] += 1
				return
			pos = j
		self.seen[c] += 1
		self.X[c * self.slots + pos].copy_(x16, non_blocking=True)

	def sample(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
		"""Draw a class-balanced sample: classes uniform over non-empty classes,
		slots uniform within each chosen class. Returns (class_idx, slot_idx)."""
		nonempty = np.where(self.filled > 0)[0]
		cls = rng.choice(nonempty, size=n, replace=True)
		slots = (rng.random(n) * self.filled[cls]).astype(np.int64)
		return cls, slots

	def gather(self, cls: np.ndarray, slots: np.ndarray) -> torch.Tensor:
		"""Gather rows as an fp16 tensor on the reservoir's device."""
		idx = torch.from_numpy(cls * self.slots + slots)
		if self.X.is_cuda:
			idx = idx.to(self.X.device)
		return torch.index_select(self.X, 0, idx)


class _StreamingFeatureStats:
	"""Streaming global feature mean/variance over all training states.

	Used to standardize inputs for the SGD probe.  Memory: sum + sumsq
	(2 x n_layers x state_dim) fp32 = ~0.46 GB at 1M-dim x 56 layers.
	"""

	def __init__(self, n_layers: int, state_dim: int, device: str):
		self.sum = torch.zeros(n_layers, state_dim, dtype=torch.float32, device=device)
		self.sumsq = torch.zeros(n_layers, state_dim, dtype=torch.float32, device=device)
		self.count = 0

	@property
	def total_count(self) -> int:
		return self.count

	def update(self, x: torch.Tensor) -> None:
		"""Accumulate one state (n_layers, k), any float dtype."""
		xf = x.float()
		self.sum.add_(xf)
		self.sumsq.addcmul_(xf, xf)
		self.count += 1

	def feature_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Global per-feature (mean, variance), fp32 (n_layers, k)."""
		n = float(self.count)
		mean = self.sum / n
		var = self.sumsq / n - mean * mean
		return mean, var.clamp_min_(0.0)


class StateProbing(Experiment):
	"""Per-layer linear probes on the SSM recurrent state for target profession."""

	NUM_SAMPLES        = 15000
	TRAIN_FRACTION     = 0.8
	LR                 = 1e-3
	WEIGHT_DECAY_RAW   = 0.0    # L2 with weak gradient signal enforces collapse
	WEIGHT_DECAY_JL    = 1e-4
	NUM_PROFESSIONS    = 20
	PROJECTION_DIM     = 4096   # JL target dimension k; original d = 1,048,576
	PROJECTION_SEED    = 0      # fixed seed for reproducible JL matrix R

	# ---- Forward batching (Fix: B=1 forwards dominated wall-clock) ----
	MAMBA_FWD_BATCH = 12  # prompts stepped together through the model

	# ---- Online training knobs ----
	PROBE_STEPS_PER_FWD_BATCH   = 16    # SGD steps after each forward batch
	RAW_MAX_CAPTURES_PER_PROMPT = 64   # per-prompt capture subsample (raw only)
	NORM_WARMUP_STATES          = 4096 # states before freezing feature mean/std
	                                    # (SGD training starts after the freeze)

	# ---- Class-balanced reservoir sizes ----
	# Raw: each slot is (56 x 1M) fp16 ~= 117 MB -> ~2.35 GB per slot across
	# 20 classes.  Default 24 slots ~= 56 GB CPU RAM; sbatch --mem=120g is
	# enough with headroom for the model-checkpoint CPU load peak (~40 GB).
	# Override with --state-probing-reservoir-slots (64 slots ~= 150 GB
	# needs ~200 GB --mem).
	# JL:  20 x 512 x (56 x 4096) fp16 = ~4.7 GB -> probe GPU.
	RES_SLOTS_PER_CLASS_RAW = 24
	RES_SLOTS_PER_CLASS_JL  = 512

	# ---- SGD minibatch sizes ----
	# Raw gradient batches are gathered from the CPU reservoir in micro-batches
	# of RAW_GRAD_MICRO_BATCH to bound both the H2D staging and the fp16+fp32
	# transient (a chunk's .float() briefly holds both dtypes at once).
	BATCH_SIZE_RAW       = 64
	BATCH_SIZE_JL        = 1024
	RAW_GRAD_MICRO_BATCH = 8

	# ------------------------------------------------------------------ config

	@property
	def _projection_mode(self) -> str:
		return self.args.state_projection

	@property
	def _projection_dim(self) -> int:
		return int(self.args.state_projection_dim)

	@property
	def _num_gpus(self) -> int:
		return int(self.args.state_probing_num_gpus)

	@property
	def _probe_device(self) -> str:
		"""Device hosting the ProbeBank, optimizer, feature stats, staging."""
		if self._num_gpus >= 2:
			return "cuda:1"
		return self.device

	@property
	def _weight_decay(self) -> float:
		return self.WEIGHT_DECAY_JL if self._projection_mode == "jl" else self.WEIGHT_DECAY_RAW

	@property
	def _reservoir_slots(self) -> int:
		override = self.args.state_probing_reservoir_slots
		if override is not None:
			return int(override)
		return self.RES_SLOTS_PER_CLASS_JL if self._projection_mode == "jl" else self.RES_SLOTS_PER_CLASS_RAW

	@staticmethod
	def _reservoir_bytes(n_layers: int, state_dim: int, slots_per_class: int,
	                     num_classes: int = NUM_PROFESSIONS) -> int:
		return int(num_classes * slots_per_class * n_layers * state_dim * 2)

	@property
	def _reservoir_device(self) -> str:
		# Raw states are far too large for GPU residency -> CPU RAM.
		return self._probe_device if self._projection_mode == "jl" else "cpu"

	@property
	def _batch_size(self) -> int:
		return self.BATCH_SIZE_JL if self._projection_mode == "jl" else self.BATCH_SIZE_RAW

	@property
	def _micro_batch(self) -> int:
		if self._projection_mode == "jl":
			return self._batch_size
		return min(self.RAW_GRAD_MICRO_BATCH, self._batch_size)

	# ------------------------------------------------------------------ paths

	@property
	def _output_path_extra(self) -> tuple[str, ...]:
		return (self._run_sha,)

	def _resolve_run_sha(self) -> str:
		"""Return the run SHA used for raw_results/plots subdirectories.

		With --skip-collection, --run-sha must point at an existing collection run.
		Otherwise a fresh SHA is generated for a new collection run.
		"""
		run_sha = self.args.run_sha
		if self.args.skip_collection:
			if not run_sha:
				raise ValueError(
					"--skip-collection requires --run-sha (the SHA printed at the start of the collection run)"
				)
			return run_sha
		if run_sha:
			raise ValueError("--run-sha is only valid together with --skip-collection")
		return uuid.uuid4().hex[:4]

	# ------------------------------------------------------------------ run

	def run(self):
		if self.args.model not in MAMBA_MODELS:
			raise ValueError(f"state_probing is only supported for mamba models: {MAMBA_MODELS}")
		if self.args.dataset not in ("profbook", "profbook_local"):
			raise ValueError(
				"state_probing is only supported for --dataset profbook or profbook_local"
			)
		if self._num_gpus >= 2 and torch.cuda.device_count() < 2:
			raise ValueError(
				f"--state-probing-num-gpus=2 requires at least 2 visible CUDA "
				f"devices, but only {torch.cuda.device_count()} are visible."
			)
		if self._projection_mode == "none" and self._num_gpus < 2:
			raise ValueError(
				"Raw (unprojected) probing does not fit on a single 45 GB GPU: "
				"the Mamba-8B backbone (~17 GB) + probe bank & Adam (~17.5 GB) "
				"exceed the budget. Use "
				"--state-probing-num-gpus 2, or --state-projection jl."
			)

		self._run_sha = self._resolve_run_sha()
		print(f"Run SHA: {self._run_sha}")
		proj_desc = (
			f"JL {self._projection_dim}d" if self._projection_mode == "jl" else "none"
		)
		device_desc = (
			f"model={self.device} probes={self._probe_device}"
			if self._num_gpus >= 2 else f"single device={self.device}"
		)
		print("#" * 80)
		print(
			f"{'#' * 8} STATE PROBING (ONLINE, BALANCED+SGD) — "
			f"SAMPLES: {self.NUM_SAMPLES}  PROJ: {proj_desc}  GPUS: {device_desc} "
			f"{'#' * 8}"
		)
		print("#" * 80)

		self.model.eval()
		raw_path = self._get_raw_results_path()

		_, per_layer_acc, meta = self._collect_train_eval(raw_path)

		self.results = {
			"run_sha":                     self._run_sha,
			"num_samples":                 self.NUM_SAMPLES,
			"entries_per_prompt":          self.args.state_probing_entries_per_prompt,
			"train_fraction":              self.TRAIN_FRACTION,
			"lr":                          self.LR,
			"weight_decay":                self._weight_decay,
			"mamba_fwd_batch":             self.MAMBA_FWD_BATCH,
			"probe_steps_per_fwd_batch":   self.PROBE_STEPS_PER_FWD_BATCH,
			"batch_size":                  self._batch_size,
			"micro_batch":                 self._micro_batch,
			"raw_max_captures_per_prompt": self.RAW_MAX_CAPTURES_PER_PROMPT,
			"reservoir_slots_per_class":   self._reservoir_slots,
			"reservoir_device":            self._reservoir_device,
			"norm_warmup_states":          self.NORM_WARMUP_STATES,
			"num_gpus":                    self._num_gpus,
			"model_device":                self.device,
			"probe_device":                self._probe_device,
			"projection":                  "gaussian_jl" if self._projection_mode == "jl" else "none",
			"projection_dim":              meta.get("projection_dim"),
			"original_dim":                meta.get("original_dim"),
			"projection_seed":             self.PROJECTION_SEED if self._projection_mode == "jl" else None,
			"per_layer_acc":               per_layer_acc,
			"results_df_path":             meta["results_df_path"],
		}
		print(f"Run SHA: {self._run_sha}")

	def save_plots(self):
		df_path = self.results.get("results_df_path", "")
		if not df_path or not os.path.exists(df_path):
			print(f"No results dataframe found at: {df_path}")
			return
		df = pd.read_parquet(df_path)
		plots_path = self._get_plots_path()
		self._plot_accuracy_vs_layer(df, plots_path, "probe_success", "sgd")
		self._plot_accuracy_vs_profession(df, plots_path, "probe_success", "sgd")
		self._plot_heatmap(df, plots_path, "probe_success", "sgd")

	# ------------------------------------------------------------------ helpers

	def _empty_cuda_cache(self) -> None:
		"""Release cached CUDA memory on every device actually in use."""
		if not torch.cuda.is_available():
			return
		device_indices = {0}
		if self._num_gpus >= 2:
			device_indices.add(1)
		for idx in device_indices:
			with torch.cuda.device(idx):
				torch.cuda.empty_cache()

	def _cuda_synchronize(self) -> None:
		"""Block until all queued CUDA work on the model and probe devices finishes."""
		if not torch.cuda.is_available():
			return
		devices = {self.device, self._probe_device}
		for dev in devices:
			with torch.cuda.device(dev):
				torch.cuda.synchronize()

	def _build_jl_matrix(self, original_dim: int, k: int) -> torch.Tensor:
		"""Build JL projection matrix R (original_dim, k) on GPU in bfloat16.

		Entries drawn from N(0, 1/k) in float32 column blocks of 512 to avoid
		allocating the full (d x k) float32 tensor (~16 GB) before casting.
		"""
		print("Building JL projection matrix R on GPU ...", flush=True)
		BLOCK = 512
		R = torch.empty(original_dim, k, dtype=torch.bfloat16, device=self.device)
		gen = torch.Generator(device=self.device)
		gen.manual_seed(self.PROJECTION_SEED)
		std = 1.0 / (k ** 0.5)
		for col_start in range(0, k, BLOCK):
			col_end = min(col_start + BLOCK, k)
			block = torch.empty(original_dim, col_end - col_start,
			                    dtype=torch.float32, device=self.device)
			block.normal_(mean=0.0, std=std, generator=gen)
			R[:, col_start:col_end] = block.to(torch.bfloat16)
			del block
		print(
			f"R built: shape={tuple(R.shape)}  "
			f"dtype={R.dtype}  mem={R.numel() * 2 / 1e9:.1f} GB",
			flush=True,
		)
		return R

	def _iter_prompt_states_batched(
		self,
		dataset,
		target_idxs: List[int],
		n_layers: int,
		R,                                     # (original_dim, k) bf16 on model device, or None
		keep_caps: Optional[List[Optional[Set[int]]]] = None,
	) -> Generator[Tuple[int, _CapturePoint, int, torch.Tensor], None, None]:
		"""Token-by-token Mamba forward for a batch of prompts stepped together.

		Yields (batch_pos, capture_point, profession_idx, state_row) where
		state_row is (n_layers, k) bfloat16 on the model device.

		keep_caps[b], if not None, is the set of per-prompt capture indices to
		extract for prompt b; captures outside it are skipped without paying
		the (n_layers x 1M) stack/projection cost.

		Batching correctness: Mamba is causal and per-batch-element, so batch
		element b's recurrent state after token t depends only on its own
		tokens 0..t.  Prompts are right-padded to the batch max length; every
		capture position is inside its own prompt, so padding is only ever
		consumed after all of a prompt's captures are done.

		Must be iterated inside torch.inference_mode().
		"""
		B = len(target_idxs)
		prompts: List[str] = []
		caps_all: List[List[_CapturePoint]] = []
		profs: List[int] = []

		for tidx in target_idxs:
			dataset.target_entry_index = tidx
			batch = dataset[0]
			prompt       = batch["input"][0]
			prompt_parts = batch["prompt_parts"][0]
			label_str    = batch["label"][0]
			profs.append(dataset.professions.index(label_str))
			caps = self._profbook_capture_positions(prompt_parts, tidx)
			n_expected = self.args.state_probing_entries_per_prompt - tidx
			assert len(caps) == n_expected, (
				f"Expected {n_expected} capture points, got {len(caps)}"
			)
			prompts.append(prompt)
			caps_all.append(caps)

		# Encode each prompt separately (no cross-prompt padding side effects),
		# then right-pad into a single (B, T_max) batch.
		ids_list = []
		for p in prompts:
			ids, _ = self._encode([p])   # (1, T_i) on model device
			ids_list.append(ids[0])
		T_max = max(int(x.shape[0]) for x in ids_list)
		input_ids = torch.zeros(B, T_max, dtype=torch.long, device=self.device)
		for b, ids in enumerate(ids_list):
			input_ids[b, :ids.shape[0]] = ids

		last_pos = max(caps[-1].token_position for caps in caps_all)

		inference_params = InferenceParams(max_seqlen=T_max, max_batch_size=B)
		inference_params.key_value_memory_dict = (
			self.model.allocate_inference_cache(B, T_max, dtype=torch.bfloat16)
		)

		ptrs = [0] * B
		for t in range(last_pos + 1):
			inference_params.seqlen_offset = t
			self.model(input_ids[:, t:t + 1], inference_params=inference_params)

			for b in range(B):
				caps = caps_all[b]
				if ptrs[b] >= len(caps) or caps[ptrs[b]].token_position != t:
					continue
				ci = ptrs[b]
				ptrs[b] += 1
				if keep_caps is not None and keep_caps[b] is not None and ci not in keep_caps[b]:
					continue

				# Per-layer SSM recurrent states of batch element b:
				# (n_layers, original_dim) bfloat16
				S = torch.stack([
					inference_params.key_value_memory_dict[l][1][b].reshape(-1)
					for l in range(n_layers)
				])
				state_row = (S @ R) if R is not None else S   # (n_layers, k)

				yield b, caps[ci], profs[b], state_row

				del S, state_row

		del inference_params
		self._empty_cuda_cache()

	def _raw_keep_set(self, target_idx: int) -> Optional[Set[int]]:
		"""Evenly-spaced capture subsample for raw mode (None = keep all)."""
		n_caps = self.args.state_probing_entries_per_prompt - target_idx
		if self._projection_mode != "none" or n_caps <= self.RAW_MAX_CAPTURES_PER_PROMPT:
			return None
		return set(
			np.round(
				np.linspace(0, n_caps - 1, self.RAW_MAX_CAPTURES_PER_PROMPT)
			).astype(int).tolist()
		)

	def _probe_grad_step(
		self,
		reservoir: _BalancedReservoir,
		mu: torch.Tensor,
		sigma: torch.Tensor,
		batch_size: int,
		micro_batch: int,
		n_layers: int,
		probe_bank: _ProbeBank,
		optimizer: torch.optim.Optimizer,
		criterion: nn.Module,
		probe_device: str,
		rng: np.random.Generator,
	) -> float:
		"""One SGD update on a class-balanced minibatch of standardized states.

		The batch is gathered from the reservoir (CPU in raw mode) in
		gradient-accumulation chunks of `micro_batch`: bounds both the H2D
		staging and the fp16+fp32 cast transient. Chunk losses are scaled by
		chunk_frac so accumulated grads equal one full-batch mean backward.
		"""
		cls, slots = reservoir.sample(batch_size, rng)

		optimizer.zero_grad(set_to_none=True)
		total_loss = 0.0
		for start in range(0, batch_size, micro_batch):
			c_chunk = cls[start:start + micro_batch]
			s_chunk = slots[start:start + micro_batch]
			chunk_frac = len(c_chunk) / batch_size

			x16 = reservoir.gather(c_chunk, s_chunk)          # fp16, reservoir device
			xb = x16.to(probe_device, non_blocking=True).float()  # (chunk, L, k) fp32
			xb = (xb - mu) / sigma                             # standardized
			yb = torch.from_numpy(c_chunk.astype(np.int64)).to(probe_device)

			logits      = probe_bank(xb)                       # (chunk, L, C)
			logits_flat = logits.reshape(-1, self.NUM_PROFESSIONS)
			yb_flat     = yb.unsqueeze(1).expand(-1, n_layers).reshape(-1)
			loss = criterion(logits_flat, yb_flat) * chunk_frac
			loss.backward()

			total_loss += float(loss.item())
			del x16, xb, yb, logits, logits_flat, yb_flat, loss

		optimizer.step()
		if str(probe_device).startswith("cuda"):
			torch.cuda.synchronize(probe_device)
		return total_loss

	# ------------------------------------------------------------------ online fused phase

	def _collect_train_eval(self, raw_path: str):
		"""Online fused collection + training + evaluation (no disk state writes).

		Returns (df, per_layer_acc, meta).
		"""
		n_layers     = int(self.model.config.n_layer)
		mixer0       = self.model.backbone.layers[0].mixer
		original_dim = int(mixer0.nheads * mixer0.headdim * mixer0.d_state)

		projection_mode = self._projection_mode
		if projection_mode == "jl":
			k = self._projection_dim
		elif projection_mode == "none":
			k = original_dim
		else:
			raise ValueError(f"Unknown --state-projection: {projection_mode}")

		print(
			f"n_layers={n_layers}  original_dim={original_dim}  "
			f"projection={projection_mode}  state_dim={k}",
			flush=True,
		)

		# ---- JL matrix (model device: projection happens before any transfer) ----
		R = self._build_jl_matrix(original_dim, k) if projection_mode == "jl" else None
		if R is None:
			print("No projection: using raw states.", flush=True)

		probe_device = self._probe_device
		if probe_device != self.device:
			print(
				f"Dual-GPU mode: model on {self.device}, probes on {probe_device}",
				flush=True,
			)

		skip_collection = self.args.skip_collection
		ckpt_path = os.path.join(raw_path, "train_checkpoint.pt")

		# ---- SGD ProbeBank ----
		# Created either way: trained below, or loaded from a checkpoint if
		# --skip-collection is set.
		probe_bank = _ProbeBank(n_layers, k, self.NUM_PROFESSIONS).to(
			probe_device, dtype=torch.float32
		)

		# ---- Prompt assignment ----
		# Every prompt i gets target_idx = i % entries_per_prompt so target
		# positions are covered uniformly; seeded shuffle then 80/20 split
		# (a prompt's states never cross the train/eval boundary).
		target_indices: List[int] = [
			i % self.args.state_probing_entries_per_prompt for i in range(self.NUM_SAMPLES)
		]
		rng_np = np.random.default_rng(int(self.args.seed))
		perm = rng_np.permutation(self.NUM_SAMPLES).tolist()
		n_train            = int(self.NUM_SAMPLES * self.TRAIN_FRACTION)
		train_prompt_order = perm[:n_train]
		eval_prompt_order  = perm[n_train:]

		self.args.eval_batch_size  = 1
		self.args.eval_num_batches = 1
		dataset = self._get_dataset(self.args.state_probing_entries_per_prompt)

		FWD = self.MAMBA_FWD_BATCH
		train_batches = [
			train_prompt_order[i:i + FWD] for i in range(0, len(train_prompt_order), FWD)
		]
		eval_batches = [
			eval_prompt_order[i:i + FWD] for i in range(0, len(eval_prompt_order), FWD)
		]

		if skip_collection:
			# --------------------------------------------------------- RESUME
			# Skip the entire (multi-hour) online-training phase and load a
			# previously-trained probe bank from disk. Use this to re-run eval
			# after an eval-time crash without repeating training.
			if not os.path.exists(ckpt_path):
				raise FileNotFoundError(
					f"--skip-collection was set but no checkpoint was found at "
					f"{ckpt_path}. It is written automatically at the end of the "
					f"training phase of a normal run."
				)
			print(f"--skip-collection: loading checkpoint from {ckpt_path} ...",
			      flush=True)
			ckpt = torch.load(ckpt_path, map_location=probe_device)
			probe_bank.load_state_dict(ckpt["probe_bank_state"])
			mu    = ckpt["mu"].to(probe_device)
			sigma = ckpt["sigma"].to(probe_device)

			# Phonebook contents are a sequential Faker/`random` stream, not
			# indexed by prompt slot. A full run draws one sample per train
			# prompt before eval; skip that many draws so eval sees the same
			# names/professions as the original (and as a same-seed
			# profbook_local run).
			n_drain = len(train_prompt_order)
			print(
				f"--skip-collection: draining {n_drain} train samples from "
				f"the dataset RNG so eval matches a full run ...",
				flush=True,
			)
			for _ in range(n_drain):
				dataset.generate_sample()
		else:
			# ---- Class-balanced reservoir (CPU RAM first, while headroom is largest) ----
			slots = self._reservoir_slots
			res_gib = self._reservoir_bytes(n_layers, k, slots) / (1024 ** 3)
			print(
				f"Reservoir plan: {self.NUM_PROFESSIONS} classes x {slots} slots "
				f"on {self._reservoir_device} (~{res_gib:.1f} GiB fp16). "
				f"Recommend sbatch --mem >= {int(res_gib + 55)}g for raw mode.",
				flush=True,
			)
			gc.collect()
			self._empty_cuda_cache()
			reservoir = _BalancedReservoir(
				self.NUM_PROFESSIONS, slots, n_layers, k, self._reservoir_device,
			)

			# ---- Fused Adam ----
			# fused=True: single-kernel Adam update, avoiding large `foreach`
			# elementwise temporaries (a full-size copy of W) of the default path.
			use_fused_adam = str(probe_device).startswith("cuda")
			optimizer = torch.optim.Adam(
				probe_bank.parameters(), lr=self.LR, weight_decay=self._weight_decay,
				fused=use_fused_adam,
			)
			criterion = nn.CrossEntropyLoss()

			# ---- Streaming feature stats (provides mean/std for standardization) ----
			feat_stats = _StreamingFeatureStats(n_layers, k, probe_device)

			batch_size  = self._batch_size
			micro_batch = self._micro_batch
			print(
				f"reservoir: {self.NUM_PROFESSIONS}x{slots} on "
				f"{self._reservoir_device}  batch_size={batch_size}  "
				f"micro_batch={micro_batch}  steps/fwd_batch={self.PROBE_STEPS_PER_FWD_BATCH}  "
				f"weight_decay={self._weight_decay}",
				flush=True,
			)

			# ============================================================ TRAIN
			probe_bank.train()
			mu: Optional[torch.Tensor]    = None
			sigma: Optional[torch.Tensor] = None
			mu_cpu: Optional[torch.Tensor]    = None
			sigma_cpu: Optional[torch.Tensor] = None
			total_grad_steps   = 0
			running_loss_sum   = 0.0
			running_loss_count = 0
			prompts_done       = 0
			LOG_EVERY_PROMPTS  = 200

			chance_loss = float(np.log(self.NUM_PROFESSIONS))
			print(
				f"Training on {len(train_prompt_order)} prompts "
				f"({len(train_batches)} forward batches of {FWD}) ...",
				flush=True,
			)
			print(
				f"  random-chance CE loss (20 balanced classes) = ln(20) = {chance_loss:.4f}",
				flush=True,
			)
			for fwd_num, chunk in enumerate(tqdm(train_batches, desc="Train (online)")):
				tidxs = [target_indices[s] for s in chunk]
				keep_caps = [self._raw_keep_set(t) for t in tidxs]

				# --- Batched Mamba forward + streaming capture consumption ---
				try:
					with torch.inference_mode():
						for _, _, prof, state_row in self._iter_prompt_states_batched(
							dataset, tidxs, n_layers, R, keep_caps
						):
							x1 = state_row.to(probe_device, non_blocking=True)  # bf16
							feat_stats.update(x1)
							reservoir.insert(x1.to(torch.float16), prof, rng_np)
							del x1
					# Force any async CUDA error from this batch's forward pass to
					# surface here (with fwd_num/chunk in the log) instead of it
					# silently resurfacing much later at an unrelated CUDA call
					# (e.g. the start of eval), which makes the real culprit batch
					# impossible to identify.
					torch.cuda.synchronize(self.device)
				except RuntimeError as e:
					print(
						f"CUDA error during training fwd_num={fwd_num} "
						f"(prompt indices {chunk}): {e}",
						flush=True,
					)
					raise

				# Ensure async reservoir writes (GPU->CPU in raw mode) finish before
				# the CPU-side gather in _probe_grad_step.
				if str(probe_device).startswith("cuda"):
					torch.cuda.synchronize(probe_device)

				# --- Freeze feature standardization after warmup ---
				if mu is None and feat_stats.total_count >= self.NORM_WARMUP_STATES:
					mean, var = feat_stats.feature_stats()
					mu    = mean
					sigma = var.sqrt().clamp_min(1e-3)
					if str(probe_device).startswith("cuda"):
						torch.cuda.synchronize(probe_device)
					mu_cpu    = mu.detach().cpu()
					sigma_cpu = sigma.detach().cpu()
					print(
						f"  [norm] frozen feature mean/std from {feat_stats.total_count} states "
						f"(median sigma={float(sigma.median()):.4g})",
						flush=True,
					)

				# --- SGD probe steps on balanced standardized minibatches ---
				# Wait until every class has at least one reservoir slot so balanced
				# sampling is not silently missing labels.
				all_classes_seen = int((reservoir.filled > 0).sum()) >= self.NUM_PROFESSIONS
				if mu is not None and all_classes_seen and reservoir.total_filled >= batch_size:
					for _ in range(self.PROBE_STEPS_PER_FWD_BATCH):
						loss_val = self._probe_grad_step(
							reservoir, mu, sigma, batch_size, micro_batch,
							n_layers, probe_bank, optimizer, criterion,
							probe_device, rng_np,
						)
						running_loss_sum   += loss_val
						running_loss_count += 1
						total_grad_steps   += 1

				prompts_done += len(chunk)
				if prompts_done % LOG_EVERY_PROMPTS < FWD:
					avg_loss = running_loss_sum / max(running_loss_count, 1)
					print(
						f"  [train] prompt {prompts_done}/{len(train_prompt_order)}"
						f"  grad_steps={total_grad_steps}  avg_loss={avg_loss:.4f}"
						f"  reservoir={reservoir.total_filled}  norm_states={feat_stats.total_count}",
						flush=True,
					)
					running_loss_sum   = 0.0
					running_loss_count = 0

			# Fallback for tiny debug runs that never crossed the warmup threshold.
			if mu is None:
				mean, var = feat_stats.feature_stats()
				mu    = mean
				sigma = var.sqrt().clamp_min(1e-3)
				if str(probe_device).startswith("cuda"):
					torch.cuda.synchronize(probe_device)
				mu_cpu    = mu.detach().cpu()
				sigma_cpu = sigma.detach().cpu()

			# ---- Free training-only memory before checkpointing ----
			del optimizer, reservoir, feat_stats
			probe_bank.zero_grad(set_to_none=True)   # frees the 4.4 GB grad tensor
			self._cuda_synchronize()
			self._empty_cuda_cache()

			# ---- Checkpoint probes before eval ----
			# An eval-time crash (e.g. a CUDA error) must never lose a
			# multi-hour training run: rerun with --skip-collection --run-sha
			# to resume evaluation from this checkpoint alone.
			print(f"Saving training checkpoint to {ckpt_path} ...", flush=True)
			torch.save({
				"probe_bank_state": {
					k: v.detach().cpu()
					for k, v in probe_bank.state_dict().items()
				},
				"mu":    mu_cpu,
				"sigma": sigma_cpu,
			}, ckpt_path)

		# ================================================================ EVAL
		probe_bank.eval()

		sgd_correct = np.zeros(n_layers, dtype=np.int64)
		layer_total = np.zeros(n_layers, dtype=np.int64)
		records: List[Dict] = []

		print(
			f"Evaluating on {len(eval_prompt_order)} prompts "
			f"({len(eval_batches)} forward batches of {FWD}) ...",
			flush=True,
		)
		for eval_fwd_num, chunk in enumerate(tqdm(eval_batches, desc="Eval (online)")):
			tidxs = [target_indices[s] for s in chunk]

			try:
				with torch.inference_mode():
					for b, cp, prof, state_row in self._iter_prompt_states_batched(
						dataset, tidxs, n_layers, R, keep_caps=None
					):
						x = state_row.to(probe_device).float()      # (L, k) fp32

						# SGD probe: standardized input
						xn = (x - mu) / sigma
						sgd_preds = probe_bank(xn.unsqueeze(0))[0].argmax(dim=-1)   # (L,)

						sgd_np = sgd_preds.cpu().numpy()
						del x, xn, sgd_preds

						target_entry_idx = cp.entry_index - cp.post_target
						for l in range(n_layers):
							s_ok = int(sgd_np[l] == prof)
							sgd_correct[l] += s_ok
							layer_total[l] += 1
							records.append({
								"layer":               l,
								"target_entry_index":  target_entry_idx,
								"post_target_entries": cp.post_target,
								"profession":          prof,
								"probe_success":       s_ok,
							})
				# See the matching comment in the training loop: force any async
				# CUDA error from this batch to surface here, with context.
				self._cuda_synchronize()
			except RuntimeError as e:
				print(
					f"CUDA error during eval fwd_num={eval_fwd_num} "
					f"(prompt indices {chunk}): {e}",
					flush=True,
				)
				raise

		# Free model-side objects
		if R is not None:
			del R
		del probe_bank, mu, sigma
		self._empty_cuda_cache()

		# ================================================================ OUTPUT
		per_layer_acc: Dict[str, float] = {}
		for l in range(n_layers):
			acc_s = float(sgd_correct[l] / max(layer_total[l], 1))
			per_layer_acc[str(l)] = acc_s
			print(f"Layer {l:03d} | sgd_acc={acc_s:.4f}")

		df = pd.DataFrame(records)
		df_path = os.path.join(raw_path, "probe_results.parquet")
		df.to_parquet(df_path, index=False)
		print(f"Saved probe results to: {df_path}")

		meta = {
			"projection_dim":  k if projection_mode == "jl" else None,
			"original_dim":    original_dim,
			"results_df_path": df_path,
		}
		return df, per_layer_acc, meta

	# ------------------------------------------------------------------ capture positions

	def _profbook_capture_positions(
		self,
		prompt_parts: Dict,
		target_index: int,
	) -> List[_CapturePoint]:
		"""Find the token position of the (single) profession token for every
		entry at or after target_index in a profbook / profbook_local prompt."""
		entries: List[str] = prompt_parts["entries"]
		prefix: str        = prompt_parts["prefix"]

		out: List[_CapturePoint] = []
		prompt     = ""
		num_tokens = 0
		is_local = self.args.dataset == "profbook_local"
		flag_location = self.args.local_dataset_flag_location
		flag_entries = self.args.local_dataset_flag_entries

		def token_count(text: str) -> int:
			enc = self.tokenizer(text)
			return len(enc.input_ids) if hasattr(enc, "input_ids") else len(enc)

		def append(text: str) -> None:
			nonlocal prompt, num_tokens
			prompt     += text
			num_tokens  = token_count(prompt)

		def maybe_capture(entry_idx: int) -> None:
			if entry_idx >= target_index:
				# num_tokens is now the count AFTER the profession token,
				# so num_tokens - 1 is its 0-indexed position.
				out.append(_CapturePoint(
					token_position=num_tokens - 1,
					entry_index=entry_idx,
					post_target=entry_idx - target_index,
				))

		append(prefix)

		for entry_idx, entry_str in enumerate(entries):
			entry_body = entry_str.rstrip("\n")

			# Mirrors ProfbookEntry.__str__: whether (and which) flag word is
			# present on this entry depends on --local-dataset-flag-entries.
			# With "target" (the common case) only the target entry is
			# flagged; all other entries render as plain "Name: Profession".
			flag_word = ""
			if is_local:
				is_target = entry_idx == target_index
				if is_target and flag_entries in ("target", "target_and_skip"):
					flag_word = "target"
				elif not is_target and flag_entries in ("skip", "target_and_skip"):
					flag_word = "skip"

			if is_local and flag_location == "name_start":
				# flagged:   "target First Last: profession" / "skip First Last: profession"
				# unflagged: "First Last: profession"
				if flag_word:
					flag_prefix = flag_word + " "
					assert entry_body.startswith(flag_prefix), (
						f"Entry {entry_idx}: expected prefix {flag_prefix!r} in {entry_body!r}"
					)
					rest = entry_body[len(flag_prefix):]
					append(flag_prefix)
				else:
					rest = entry_body
				entry_name, prof = rest.split(": ", 1)
				append(entry_name)
				append(": ")
				append(prof)
			elif is_local and flag_location == "name_end":
				# flagged:   "First Last: target profession" / "First Last: skip profession"
				# unflagged: "First Last: profession"
				entry_name, rest = entry_body.split(": ", 1)
				if flag_word:
					parsed_flag_word, prof = rest.split(" ", 1)
					assert parsed_flag_word == flag_word, (
						f"Entry {entry_idx}: expected flag {flag_word!r}, got {parsed_flag_word!r}"
					)
				else:
					prof = rest
				append(entry_name)
				append(": ")
				if flag_word:
					append(flag_word + " ")
				append(prof)
			else:
				# "First Last: profession"
				name, prof = entry_body.split(": ", 1)
				append(name)
				append(": ")
				append(prof)  # profession is a single token
			maybe_capture(entry_idx)
			append("\n")

		return out

	# ------------------------------------------------------------------ phase 3

	def _plot_accuracy_vs_layer(self, df: "pd.DataFrame", plots_path: str,
	                            success_col: str, tag: str):
		layer_acc = df.groupby("layer")[success_col].mean()

		fig, ax = plt.subplots(figsize=(14, 5))
		ax.plot(layer_acc.index, layer_acc.values, marker="o", markersize=4, linewidth=1.5)
		ax.set_xlabel("Layer")
		ax.set_ylabel("Probe accuracy")
		ax.set_title(
			f"Probe Accuracy vs Layer ({tag})\n"
			f"({self.args.model}, {self.args.task}, "
			f"entries={self.args.state_probing_entries_per_prompt})"
		)
		ax.set_xticks(range(0, int(df["layer"].max()) + 1, 4))
		ax.grid(True, alpha=0.3)
		plt.tight_layout()
		out = os.path.join(plots_path, f"accuracy_vs_layer_{tag}.png")
		plt.savefig(out, dpi=150)
		plt.close()
		print(f"Saved: {out}")

	def _plot_accuracy_vs_profession(self, df: "pd.DataFrame", plots_path: str,
	                                 success_col: str, tag: str):
		prof_names_path = os.path.join(
			os.path.dirname(__file__), "..", "datasets", "professions.txt"
		)
		with open(prof_names_path) as f:
			prof_names = [l.strip() for l in f if l.strip()]

		acc_by_prof = df.groupby("profession")[success_col].mean()
		accs = [float(acc_by_prof.get(i, 0.0)) for i in range(self.NUM_PROFESSIONS)]

		fig, ax = plt.subplots(figsize=(14, 6))
		bars = ax.bar(range(self.NUM_PROFESSIONS), accs)
		ax.set_xticks(range(self.NUM_PROFESSIONS))
		ax.set_xticklabels(prof_names, rotation=45, ha="right", fontsize=9)
		ax.set_ylabel("Probe accuracy")
		ax.set_title(
			f"Probe Accuracy vs Profession ({tag})\n"
			f"({self.args.model}, {self.args.task}, "
			f"entries={self.args.state_probing_entries_per_prompt})"
		)
		ax.set_ylim(0, 1)
		ax.grid(True, alpha=0.3, axis="y")
		plt.tight_layout()
		out = os.path.join(plots_path, f"accuracy_vs_profession_{tag}.png")
		plt.savefig(out, dpi=150)
		plt.close()
		print(f"Saved: {out}")

	def _plot_heatmap(self, df: "pd.DataFrame", plots_path: str,
	                  success_col: str, tag: str):
		bucket_size = 5
		df = df.copy()
		df["target_bucket"] = (df["target_entry_index"] // bucket_size) * bucket_size
		df["post_bucket"]   = (df["post_target_entries"] // bucket_size) * bucket_size

		pivot = df.groupby(["post_bucket", "target_bucket"])[success_col].mean().unstack()

		fig, ax = plt.subplots(figsize=(14, 10))
		sns.heatmap(
			pivot,
			ax=ax,
			cmap="viridis",
			annot=False,
			linewidths=0.3,
			cbar_kws={"label": "Mean probe accuracy"},
		)

		x_labels = [str(int(c)) for c in pivot.columns]
		y_labels = [str(int(r)) for r in pivot.index]
		ax.set_xticks([i + 0.5 for i in range(len(x_labels))])
		ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
		ax.set_yticks([i + 0.5 for i in range(len(y_labels))])
		ax.set_yticklabels(y_labels, rotation=0, fontsize=8)
		ax.set_xlabel("target_entry_index (bucket start)")
		ax.set_ylabel("post_target_entries (bucket start)")
		ax.set_title(
			f"Probe Accuracy Heatmap (buckets of {bucket_size}, {tag})\n"
			f"({self.args.model}, {self.args.task}, "
			f"entries={self.args.state_probing_entries_per_prompt})"
		)
		plt.tight_layout()
		out = os.path.join(plots_path, f"accuracy_heatmap_{tag}.png")
		plt.savefig(out, dpi=150)
		plt.close()
		print(f"Saved: {out}")
