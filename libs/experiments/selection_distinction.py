"""Selection Distinction experiment.

Tracks a chosen Mamba2 mixer component at every token, grouped by phonebook
token category, to measure whether the target entry is treated differently
from distractors.

What is measured
----------------
Per (layer, head) mean of --selectivity-component over tokens in:
  target_entry_name / target_entry_number
  other_entries_name / other_entries_number
Component choices (--selectivity-component, default dt_raw):
  delta             — Δ = softplus(dt_logit + dt_bias)
  dt_raw            — dt_logit only (no bias, no softplus)
  B_norm            — L2 norm of pre-conv B, repeated to per-head
  B_norm_post_conv  — same after simulating causal conv1d + SiLU
  forget_factor     — exp(Δ · A)

Requirements
------------
Uses Mamba2 mixer internals (in_proj layout, InferenceParams). Token
categories are recovered from prompt_parts via character-span binary
search on the model's actual token ids (not incremental re-tokenization).

Configuration
-------------
  --selectivity-component
  --selection-distinction-entries-per-prompt  (default 150)
  --eval-num-batches / --eval-batch-size
  --dataset (phonebook / phonebook_local and flag location/entries)

Protocol per sample
-------------------
1. Encode the prompt; classify each token into the four categories above
   (flag words and punctuation go to rest and are not aggregated).
2. Token-by-token forward with InferenceParams; an in_proj hook records
   the chosen per-head component.
3. Accumulate sums and counts per category per layer.

Metric
------
Mean component value = sum / count (count 0 → 0). No generation scoring.

Outputs
-------
results.json: one (n_layers, n_heads) array per category.
Plots: {component}_by_layer.png, _by_head.png, _heatmap.png,
       _entry_number_head_summary_contrast.png.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from mamba_ssm.utils.generation import InferenceParams

from libs.experiments.experiment import Experiment


CATEGORIES = [
	"target_entry_name",
	"target_entry_number",
	"other_entries_name",
	"other_entries_number",
]
REST_CAT_ID = len(CATEGORIES)
CAT_NAME_TO_ID = {cat: i for i, cat in enumerate(CATEGORIES)}

# Display labels for each head-summary choice.
_ENTRY_NUMBER_HEAD_SUMMARIES = {
	"mean":   np.mean,
	"median": np.median,
	"p5":    lambda x: np.percentile(x, 5),
	"min":    np.min,
}

_ENTRY_NUMBER_HEAD_SUMMARY_LABELS = {
	"mean":   "Mean across heads",
	"median": "Median across heads",
	"p5":    "5th percentile",
	"min":    "Minimum across heads",
}

COMPONENT_LABELS = {
	'delta':             r"$\Delta$",
	'dt_raw':            r"$dt_{\mathrm{raw}}$",
	'B_norm':            r"$\|B\|_{\mathrm{pre}}$",
	'B_norm_post_conv':  r"$\|B\|_{\mathrm{post}}$",
	'forget_factor':     r"$e^{\Delta A}$",
}


@dataclass
class PromptTokenClassification:
	other_entries_name: set = field(default_factory=set)
	other_entries_number: set = field(default_factory=set)
	target_entry_name: set = field(default_factory=set)
	target_entry_number: set = field(default_factory=set)
	rest: set = field(default_factory=set)


class SelectionDistinction(Experiment):
	@property
	def _component_label(self):
		return COMPONENT_LABELS[self.args.selectivity_component]

	@property
	def _variant_prefix(self):
		return f"{self.args.selectivity_component}_"

	def _compute_component(self, in_proj_output, *, dt_bias, A, n_heads, d_ssm, ngroups, d_state,
					   conv_weight=None, conv_bias=None, conv_state=None):
		"""Compute the per-head scalar component from the raw in_proj output and learned parameters.

		Args:
			in_proj_output: (batch, [1,] d_in_proj) — full output of mixer.in_proj.
			                Layout: [z0, x0, z, xBC, dt_logit].
			dt_bias:  (n_heads,)  — learned per-head bias parameter (mixer.dt_bias).
			A:        (n_heads,)  — negative decay rates: -exp(A_log).
			n_heads:  int         — number of SSM heads.
			d_ssm:    int         — SSM feature dimension (= n_heads × headdim).
			ngroups:  int         — number of head-groups sharing B/C projections.
			d_state:  int         — SSM state dimension per group.

		  Only required for 'B_norm_post_conv':
			conv_weight: (D, d_conv) — depthwise conv1d kernel, D = d_ssm + 2*ngroups*d_state.
			conv_bias:   (D,) or None — conv1d bias.
			conv_state:  (batch, D, d_conv) — causal conv state *before* the current token
			             is inserted (read from inference_params).

		Returns: (batch, n_heads) float tensor.

		component choices:
		  'delta'             — Δ = softplus(dt_logit + dt_bias): the discretization step-size.
		  'dt_raw'            — dt_logit only (no bias, no softplus): the purely input-dependent
		                        component.
		  'B_norm'            — L2 norm of the pre-conv B projection per group, repeated to
		                        per-head. Measures token-local write-strength.
		  'B_norm_post_conv'  — Same norm but computed after simulating the causal conv1d step,
		                        reflecting the effective write-strength that actually reaches
		                        the SSM (includes temporal smoothing from the last d_conv tokens).
		  'forget_factor'     — dA = exp(Δ · A): the state retention coefficient combining the
		                        input-dependent Δ with the learned per-head decay rate A.
		"""
		choice = self.args.selectivity_component

		if choice in ('delta', 'dt_raw', 'forget_factor'):
			dt_logit = in_proj_output[..., -n_heads:].float()

		if choice == 'delta':
			return F.softplus(dt_logit + dt_bias)
		elif choice == 'dt_raw':
			return dt_logit
		elif choice == 'forget_factor':
			delta = F.softplus(dt_logit + dt_bias)
			return torch.exp(delta * A)
		elif choice in ('B_norm', 'B_norm_post_conv'):
			d_in_proj = in_proj_output.shape[-1]
			d_mlp = (d_in_proj - 2 * d_ssm - 2 * ngroups * d_state - n_heads) // 2
			xBC_start = 2 * d_mlp + d_ssm

			if choice == 'B_norm':
				B_start = xBC_start + d_ssm
				B_end = B_start + ngroups * d_state
				B_vec = in_proj_output[..., B_start:B_end].float()
			else:
				xBC_end = xBC_start + d_ssm + 2 * ngroups * d_state
				xBC_pre = in_proj_output[..., xBC_start:xBC_end].float()
				if xBC_pre.dim() == 3:
					xBC_pre = xBC_pre.squeeze(1)
				# Simulate causal conv1d: roll old state, insert current token, convolve
				rolled = torch.roll(conv_state.float(), shifts=-1, dims=-1)
				rolled[:, :, -1] = xBC_pre
				xBC_post = torch.sum(rolled * conv_weight.float(), dim=-1)
				if conv_bias is not None:
					xBC_post = xBC_post + conv_bias.float()
				xBC_post = F.silu(xBC_post)
				B_vec = xBC_post[:, d_ssm:d_ssm + ngroups * d_state]

			B_vec = B_vec.view(*B_vec.shape[:-1], ngroups, d_state)
			B_norms = B_vec.norm(dim=-1)                                    # (..., ngroups)
			return B_norms.repeat_interleave(n_heads // ngroups, dim=-1)    # (..., n_heads)
		else:
			raise ValueError(f"Unknown selectivity_component: {choice!r}")

	def run(self):
		length = self.args.selection_distinction_entries_per_prompt
		print(f"{'#' * 80}")
		print(f"{'#' * 12} SELECTIVITY INTERNALS ({self.args.selectivity_component}) — LENGTH: {length} {'#' * 12}")
		print(f"{'#' * 80}")

		self.model.eval()
		dataset = self._get_dataset(length)
		n_layers = self.model.config.n_layer
		n_heads = self.model.backbone.layers[0].mixer.nheads

		cat_component_sums = {cat: np.zeros((n_layers, n_heads)) for cat in CATEGORIES}
		cat_token_counts = {cat: np.zeros(n_layers, dtype=int) for cat in CATEGORIES}

		component_storage = {}
		hook_context = {}
		hooks = self._register_hooks(n_layers, component_storage, hook_context)

		try:
			for batch_idx in tqdm(range(self.args.eval_num_batches),
								  desc=f"Selectivity Internals ({self.args.selectivity_component}, len={length})"):
				batch = dataset[batch_idx]
				input_ids, attention_mask = self._encode(batch["input"])
				B, T = input_ids.shape
				token_cats = self._build_token_categories(
					batch["prompt_parts"], batch["target_index"], attention_mask, T,
					input_ids,
				)

				inference_params = InferenceParams(max_seqlen=T, max_batch_size=B)
				inference_params.key_value_memory_dict = (
					self.model.allocate_inference_cache(B, T, dtype=torch.bfloat16)
				)
				hook_context['inference_params'] = inference_params

				with torch.inference_mode():
					for t in range(T):
						inference_params.seqlen_offset = t
						component_storage.clear()
						self.model(input_ids[:, t : t + 1], inference_params=inference_params)

						for l in range(n_layers):
							component = component_storage[l]  # (B, n_heads)
							for cat_id, cat_name in enumerate(CATEGORIES):
								mask = token_cats[:, t] == cat_id
								if mask.any():
									cat_component_sums[cat_name][l] += component[mask].sum(dim=0).cpu().numpy()
									cat_token_counts[cat_name][l] += mask.sum().item()

				del inference_params
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
		finally:
			for h in hooks:
				h.remove()

		self.results = {
			cat: cat_component_sums[cat] / np.maximum(cat_token_counts[cat][:, None], 1)
			for cat in CATEGORIES
		}

	def _register_hooks(self, n_layers, component_storage, hook_context=None):
		"""Register forward hooks on each Mamba2 layer's in_proj to capture the per-head component.

		in_proj projects the residual stream to all SSM inputs in one shot. Its output is
		laid out as [z0, x0, z, xBC, dt_logit]:
		  z0, x0    — gated MLP branch (zero-width when d_ssm == d_inner)
		  z         — SSM gating vector; multiplied element-wise with the SSM output
		              after normalisation to control how much of the SSM result passes through
		  xBC       — concatenation of:
		                x : SSM input features (width: d_ssm)
		                B : input-dependent state-expansion matrix (width: n_groups × d_state)
		                C : input-dependent state-readout matrix   (width: n_groups × d_state)
		              all three are fed into the causal conv1d before the SSM scan
		  dt_logit  — input-dependent discretization time-step logit (width: n_heads); it becomes the
		              discretisation step-size Δ_t = softplus(dt_logit + dt_bias), where
		              dt_bias is a learned per-head parameter (mixer.dt_bias)

		Args:
			hook_context: optional mutable dict shared with the run loop. Must contain
			    'inference_params' when using 'B_norm_post_conv' (needed to read conv_state).
		"""
		hooks = []
		needs_conv = self.args.selectivity_component == 'B_norm_post_conv'

		for l in range(n_layers):
			mixer = self.model.backbone.layers[l].mixer
			dt_bias = mixer.dt_bias.detach().clone()
			A = -torch.exp(mixer.A_log.float()).detach().clone()
			n_heads = mixer.nheads
			d_ssm = mixer.d_ssm
			ngroups = mixer.ngroups
			d_state = mixer.d_state

			conv_weight = conv_bias = None
			if needs_conv:
				conv_weight = mixer.conv1d.weight.squeeze(1).detach().clone()   # (D, d_conv)
				if mixer.conv1d.bias is not None:
					conv_bias = mixer.conv1d.bias.detach().clone()              # (D,)

			def make_hook(layer_idx, dt_bias, A, n_heads, d_ssm, ngroups, d_state,
						  conv_weight, conv_bias):
				def hook_fn(module, input, output):
					conv_state = None
					if needs_conv and hook_context is not None:
						ip = hook_context['inference_params']
						conv_state = ip.key_value_memory_dict[layer_idx][0]

					component = self._compute_component(
						output.float(),
						dt_bias=dt_bias.float(), A=A, n_heads=n_heads,
						d_ssm=d_ssm, ngroups=ngroups, d_state=d_state,
						conv_weight=conv_weight, conv_bias=conv_bias,
						conv_state=conv_state,
					)
					if component.dim() == 3:
						component = component.squeeze(1)  # (batch, n_heads)
					component_storage[layer_idx] = component.detach()
				return hook_fn

			h = mixer.in_proj.register_forward_hook(
				make_hook(l, dt_bias, A, n_heads, d_ssm, ngroups, d_state,
						  conv_weight, conv_bias)
			)
			hooks.append(h)
		return hooks

	def _char_spans_to_token_ranges(
		self, ids: List[int], spans: List[Tuple[Tuple[int, int], Tuple[int, int]]]
	) -> Tuple[List[List[int]], List[List[int]]]:
		"""Map (name_span, value_span) character-offset pairs to token-index
		ranges within `ids`, the single, final tokenization of the complete
		text these offsets were recorded against.

		Ported from PatchingAndScaling._char_spans_to_token_ranges. Incremental
		re-tokenization of a growing prefix is unreliable for BPE tokenizers:
		appending ": " can leave a trailing space token, and appending the
		value then fuses that space into the first chunk (e.g. " 790"), so
		the token-count delta undercounts and the first value token is
		misclassified as rest. The same fusion hits a flagged entry's first
		name token after "target ". Locating boundaries on one complete
		tokenization via decoded-prefix-length binary search does not have
		this problem (verified: decoding growing prefixes of one fixed
		tokenization reconstructs exact text prefixes).
		"""
		decoded_len_cache: Dict[int, int] = {0: 0}

		def decoded_len(k: int) -> int:
			if k not in decoded_len_cache:
				decoded_len_cache[k] = len(self._decode(ids[:k]))
			return decoded_len_cache[k]

		def char_to_tok(char_offset: int) -> int:
			lo, hi = 0, len(ids)
			while lo < hi:
				mid = (lo + hi) // 2
				if decoded_len(mid) < char_offset:
					lo = mid + 1
				else:
					hi = mid
			return lo

		name_ranges, val_ranges = [], []
		for (name_start, name_end), (val_start, val_end) in spans:
			name_ranges.append(list(range(char_to_tok(name_start), char_to_tok(name_end))))
			val_ranges.append(list(range(char_to_tok(val_start), char_to_tok(val_end))))
		return name_ranges, val_ranges

	def _build_token_categories(self, prompt_parts_batch, target_indices, attention_mask, seq_len, input_ids):
		B = len(prompt_parts_batch)
		token_cats = torch.full((B, seq_len), REST_CAT_ID, dtype=torch.long)

		for s in range(B):
			ids = input_ids[s, attention_mask[s].bool()].tolist()
			classification = self._prompt_token_classification(
				prompt_parts_batch[s], target_indices[s], ids,
			)
			pad_offset = seq_len - attention_mask[s].sum().item()

			for cat_name, cat_id in CAT_NAME_TO_ID.items():
				for idx in getattr(classification, cat_name):
					token_cats[s, idx + pad_offset] = cat_id

		return token_cats

	def _iter_entry_token_ranges(self, prompt_parts, target_entry_index, ids):
		"""Yield (is_target, name_token_range, value_token_range) for each entry.

		Supports all phonebook entry formats:
		  - phonebook:                  "Name: Phone\\n"
		  - phonebook_local name_start: "target Name: Phone\\n" / "skip Name: Phone\\n"
		  - phonebook_local name_end:   "Name: target Phone\\n" / "Name: skip Phone\\n"

		The "target" / "skip" flag tokens are never included in the name or
		value ranges, consistent with other structural tokens such as ":"
		and "\\n".

		Token boundaries come from `ids` (the sequence the model actually
		saw). Character spans of each entry's name and value are mapped onto
		those ids via `_char_spans_to_token_ranges`. The value span starts
		at the space preceding the value, matching PatchingAndScaling: BPE and
		SentencePiece both fuse that space into the value's first token, so
		the only clean token boundary is immediately before the space.
		"""
		prefix = prompt_parts["prefix"]
		entries = prompt_parts["entries"]
		suffix = prompt_parts["suffix"]

		is_local = self.args.dataset == "phonebook_local"
		flag_location = self.args.local_dataset_flag_location
		flag_entries = self.args.local_dataset_flag_entries

		# Pass 1: rebuild the prompt text and record character spans.
		# No tokenization happens here — see _char_spans_to_token_ranges.
		prompt = prefix
		# Each entry: (is_target, (name_start, name_end), (val_start, val_end)).
		entry_spans: List[Tuple[bool, Tuple[int, int], Tuple[int, int]]] = []

		for entry_idx, entry in enumerate(entries):
			is_target = entry_idx == target_entry_index
			entry_body = entry.rstrip('\n')

			# Mirrors PhonebookEntry.__str__: whether (and which) flag word is
			# present on this entry depends on --local-dataset-flag-entries.
			# With "target" (the common case) only the target entry is
			# flagged; all other entries render as plain "Name: Phone".
			flag_word = ""
			if is_local:
				if is_target and flag_entries in ('target', 'target_and_skip'):
					flag_word = 'target'
				elif not is_target and flag_entries in ('skip', 'target_and_skip'):
					flag_word = 'skip'

			if is_local and flag_location == 'name_start':
				# flagged:   "target First Last: 1234567890" / "skip First Last: 1234567890"
				# unflagged: "First Last: 1234567890"
				# The flag word (if present) precedes the name; route it to rest.
				if flag_word:
					flag_prefix = flag_word + ' '
					assert entry_body.startswith(flag_prefix), (
						f"Entry {entry_idx}: expected prefix {flag_prefix!r} in {entry_body!r}"
					)
					remainder = entry_body[len(flag_prefix):]
					# Flag word itself is rest. The trailing space is bundled
					# with the name (same reason PatchingAndScaling bundles the
					# space after ":" with the value): BPE fuses it into the
					# name's first token (e.g. " Bob"), and the only clean
					# token boundary is immediately before that space.
					prompt += flag_word
					name_start = len(prompt)
					prompt += ' '
				else:
					remainder = entry_body
					name_start = len(prompt)
				entry_name, entry_number = remainder.split(': ', 1)
				prompt += entry_name
				name_end = len(prompt)
				prompt += ':'
				val_start = len(prompt)
				prompt += ' ' + entry_number
				val_end = len(prompt)
			elif is_local and flag_location == 'name_end':
				# flagged:   "First Last: target 1234567890" / "First Last: skip 1234567890"
				# unflagged: "First Last: 1234567890"
				# The flag word (if present) precedes the number; route it to rest.
				entry_name, rest = entry_body.split(': ', 1)
				if flag_word:
					parsed_flag_word, entry_number = rest.split(' ', 1)
					assert parsed_flag_word == flag_word, (
						f"Entry {entry_idx}: expected flag {flag_word!r}, got {parsed_flag_word!r}"
					)
				else:
					entry_number = rest
				name_start = len(prompt)
				prompt += entry_name
				name_end = len(prompt)
				prompt += ':'
				if flag_word:
					prompt += ' ' + flag_word
				val_start = len(prompt)
				prompt += ' ' + entry_number
				val_end = len(prompt)
			else:
				# Original phonebook: "First Last: 1234567890"
				entry_name, entry_number = entry_body.split(': ', 1)
				name_start = len(prompt)
				prompt += entry_name
				name_end = len(prompt)
				prompt += ':'
				val_start = len(prompt)
				prompt += ' ' + entry_number
				val_end = len(prompt)

			prompt += '\n'
			entry_spans.append((is_target, (name_start, name_end), (val_start, val_end)))

		prompt += suffix

		# Pass 2: map recorded character offsets onto the actual token ids.
		name_ranges, val_ranges = self._char_spans_to_token_ranges(
			ids, [(name_span, val_span) for _, name_span, val_span in entry_spans]
		)
		for (is_target, _, _), name_range, val_range in zip(entry_spans, name_ranges, val_ranges):
			yield is_target, name_range, val_range

	def _prompt_token_classification(self, prompt_parts, target_entry_index, ids) -> PromptTokenClassification:
		"""Classify each token position into name / number / rest categories."""
		output = PromptTokenClassification()
		for is_target, name_range, val_range in self._iter_entry_token_ranges(
			prompt_parts, target_entry_index, ids,
		):
			name_set = output.target_entry_name if is_target else output.other_entries_name
			number_set = output.target_entry_number if is_target else output.other_entries_number
			name_set.update(name_range)
			number_set.update(val_range)
		return output

	def save_plots(self):
		self.plot_by_layer()
		self.plot_by_head()
		self.plot_heatmap()
		self.plot_entry_number_head_summary_contrast_by_layer()

	def _style_map(self):
		return {
			"target_entry_name":    {"color": "#d62728", "ls": "-", "label": "Target Name"},
			"other_entries_name":   {"color": "#1f77b4", "ls": "-", "label": "Other Name"},
			"target_entry_number":  {"color": "#d62728", "ls": "-", "label": "Target Number"},
			"other_entries_number": {"color": "#1f77b4", "ls": "-", "label": "Other Number"},
		}

	def _model_tag(self):
		return (
			f"{self.args.model}, {self.args.task}, "
			f"length={self.args.selection_distinction_entries_per_prompt}"
		)

	def plot_by_layer(self):
		"""Head-averaged component per layer — Names on top, Numbers on bottom."""
		res = self.results
		style_map = self._style_map()
		layers = np.arange(np.asarray(res["target_entry_name"]).shape[0])

		fig, (ax_name, ax_num) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

		for ax, target_key, other_key, subtitle in [
			(ax_name, "target_entry_name",   "other_entries_name",   "Names"),
			(ax_num,  "target_entry_number", "other_entries_number", "Numbers"),
		]:
			for key in (target_key, other_key):
				s = style_map[key]
				ax.plot(layers, np.asarray(res[key]).mean(axis=1),
						color=s["color"], linestyle=s["ls"], linewidth=2, label=s["label"])
			ax.set_ylabel(f"Mean {self._component_label} (head-averaged)")
			ax.set_title(subtitle)
			ax.legend(loc="upper right")
			ax.grid(True, alpha=0.3)

		ax_num.set_xlabel("Layer Index")
		fig.suptitle(f"{self._component_label} by Layer\n({self._model_tag()})",
					 fontsize=13, fontweight="bold")
		plt.tight_layout()
		plt.savefig(self._get_plots_path() + f"{self._variant_prefix}by_layer.png", dpi=150)
		plt.close()

	def plot_by_head(self):
		"""Layer-averaged component per head — Names on top, Numbers on bottom."""
		res = self.results
		style_map = self._style_map()
		heads = np.arange(np.asarray(res["target_entry_name"]).shape[1])

		fig, (ax_name, ax_num) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

		for ax, target_key, other_key, subtitle in [
			(ax_name, "target_entry_name",   "other_entries_name",   "Names"),
			(ax_num,  "target_entry_number", "other_entries_number", "Numbers"),
		]:
			for key in (target_key, other_key):
				s = style_map[key]
				ax.plot(heads, np.asarray(res[key]).mean(axis=0),
						color=s["color"], linestyle=s["ls"], linewidth=2, label=s["label"])
			ax.set_ylabel(f"Mean {self._component_label} (layer-averaged)")
			ax.set_title(subtitle)
			ax.legend(loc="upper right")
			ax.grid(True, alpha=0.3)

		ax_num.set_xlabel("Head Index")
		fig.suptitle(f"{self._component_label} by Head\n({self._model_tag()})",
					 fontsize=13, fontweight="bold")
		plt.tight_layout()
		plt.savefig(self._get_plots_path() + f"{self._variant_prefix}by_head.png", dpi=150)
		plt.close()

	def plot_heatmap(self):
		"""Target-minus-other contrast heatmap (n_layers × n_heads).

		Top panel: target_name - other_name per (layer, head).
		Bottom panel: target_number - other_number per (layer, head).
		"""
		res = self.results
		name_contrast   = np.asarray(res["target_entry_name"])   - np.asarray(res["other_entries_name"])
		number_contrast = np.asarray(res["target_entry_number"]) - np.asarray(res["other_entries_number"])

		abs_max = max(np.abs(name_contrast).max(), np.abs(number_contrast).max())
		vmin, vmax = -abs_max, abs_max

		fig, (ax_name, ax_num) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

		for ax, data, subtitle in [
			(ax_name, name_contrast,   "Names"),
			(ax_num,  number_contrast, "Numbers"),
		]:
			im = ax.imshow(
				data, aspect="auto", origin="lower",
				vmin=vmin, vmax=vmax, cmap="RdBu",
			)
			ax.set_xlabel("Head Index")
			ax.set_ylabel("Layer Index")
			ax.set_title(f"Target − Other — {subtitle}")

		fig.colorbar(im, ax=[ax_name, ax_num],
					 label=rf"$\mathrm{{target}} - \mathrm{{other}}$  [{self._component_label}]",
					 shrink=0.75)
		fig.suptitle(f"{self._component_label} Contrast Heatmap (Layer × Head)\n({self._model_tag()})",
					 fontsize=13, fontweight="bold")
		plt.savefig(self._get_plots_path() + f"{self._variant_prefix}heatmap.png", dpi=150)
		plt.close()

	def _entry_number_head_summary_contrast(self):
		"""For each head-summary function, compute (target − other) entry-number contrast per layer.

		Returns a dict mapping summary key → (n_layers,) float array.
		"""
		target = np.asarray(self.results["target_entry_number"])   # (n_layers, n_heads)
		other  = np.asarray(self.results["other_entries_number"])  # (n_layers, n_heads)
		contrasts = {}
		for key, fn in _ENTRY_NUMBER_HEAD_SUMMARIES.items():
			target_summary = np.array([fn(target[l]) for l in range(target.shape[0])])
			other_summary  = np.array([fn(other[l])  for l in range(other.shape[0])])
			contrasts[key] = target_summary - other_summary
		return contrasts

	def plot_entry_number_head_summary_contrast_by_layer(self):
		"""Target−other entry-number contrast per layer, plotted for four head-distribution summaries."""
		contrasts = self._entry_number_head_summary_contrast()
		n_layers = np.asarray(self.results["target_entry_number"]).shape[0]
		layers = np.arange(n_layers)

		colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]

		fig, ax = plt.subplots(figsize=(12, 5))
		for (key, contrast), color in zip(contrasts.items(), colors):
			ax.plot(layers, contrast, color=color, linestyle="--", linewidth=2,
					label=_ENTRY_NUMBER_HEAD_SUMMARY_LABELS[key])

		ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
		ax.set_xlabel("Layer Index")
		ax.set_ylabel(f"Target \u2212 Other {self._component_label} (head summary)")
		ax.set_title(
			f"Entry Number Target\u2212Other Contrast by Layer (Head Summaries)\n({self._model_tag()})",
			fontsize=13, fontweight="bold",
		)
		ax.legend(loc="upper right")
		ax.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(
			self._get_plots_path() + f"{self._variant_prefix}entry_number_head_summary_contrast.png",
			dpi=150,
		)
		plt.close()
