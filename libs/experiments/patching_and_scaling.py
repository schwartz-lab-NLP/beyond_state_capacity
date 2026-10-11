"""Patching and Scaling experiment.

Causal activation-patching on one or more Mamba2 mixer components to test whether
they are responsible for the accuracy gain observed when "local" (flag-prefixed)
prompts are used. Supports patching:
  u            — the full input representation fed into mixer.in_proj (its input,
                 of dimension d_model).
  dt_raw       — the input-dependent discretization logit (last n_heads columns of
                 the mixer.in_proj output), pre-bias, pre-softplus.
  delta        — Δ = softplus(dt_raw + dt_bias), the actual discretization
                 step-size used by the SSM scan. Implemented by capturing the
                 donor's Δ and inverting it back into dt_raw's logit space before
                 writing it into the in_proj output (donor and recipient share the
                 same dt_bias, so this reconstructs the donor's Δ exactly).
  B_pre_conv   — the input-dependent B projection, sliced from the in_proj output
                 before the causal conv1d.
  C_pre_conv   — the input-dependent C projection, sliced from the in_proj output
                 before the causal conv1d.
  B_post_conv  — B after the causal conv1d + SiLU activation, i.e. the exact
                 tensor consumed by the SSM scan. Captured/patched via a
                 monkeypatch of mamba_chunk_scan_combined, since the conv1d is
                 invoked through a fused Triton kernel rather than through the
                 conv1d nn.Module, leaving no module boundary to hook directly.
  C_post_conv  — same as B_post_conv, for C.
Select which component(s) to patch with --patching-and-scaling-components. All
selected components are patched jointly — at each swept layer-range, a single
generate call overwrites every requested component at once, giving a single
"components patched together" condition per layer-range. To compare, e.g., "dt_raw
only" vs. "u only" vs. "both together", run the experiment multiple times with
different --patching-and-scaling-components values.

Supports both dataset families:
  phonebook / phonebook_local — entries are "Name: Phone", label is a 10-digit number.
  profbook  / profbook_local  — entries are "Name: Profession", label is a profession word.
Pass --dataset with any of these four values; the experiment resolves it to the
matching "_local" dataset automatically.

The patching direction is controlled independently by --patch-donor and
--patch-recipient (each 'local' or 'global'). Donor and recipient must differ.
For example, --patch-donor local --patch-recipient global captures from the
local prompt and patches into the global prompt.

Which entry tokens are patched is controlled by --patching-and-scaling-entries:
  target_entry       — only the target entry's name/value tokens (default).
  non_target_entries — only distractor entries' tokens.
  all_entries        — every entry's name/value tokens.

Which part of each selected entry is patched is controlled by
--patching-and-scaling-entry-part:
  both   — name and value tokens (default).
  name   — only the name tokens of each selected entry.
  number — only the value tokens (phone digits / profession word).
  all    — the entire entry: name, colon, number (including its leading
           space), and trailing newline. Flag words are excluded — they
           exist only on the local prompt and have no aligned counterpart.

Alternatively, pass --patch-donor scaling to enable in-place scaling mode.
In this mode no donor prompt is used. Instead the component's own values on
the recipient prompt are multiplied by --patch-scale (default 2.0) at
target-entry token positions and divided by --patch-scale at all other
patched entries' token positions. This tests whether artificially amplifying
the component's selectivity signal on the target entry (while suppressing it
on distractors) recovers accuracy on the prompt type being studied.
The same --patching-and-scaling-entries and --patching-and-scaling-entry-part filters
apply to which positions are scaled.

Protocol per sample (standard donor mode)
------------------------------------------
1. Generate on the donor prompt and cache every requested component at every layer
   (a single generate call captures all requested components at once).
2. Build a token-position alignment from each entry's selected tokens in the local
   prompt to the corresponding positions in a flag-stripped global prompt.
3. Generate on the recipient prompt with every requested component overwritten
   together at aligned positions in a specified set of layers.
4. Record whether the output label is correct on the recipient prompt.

Protocol per sample (scaling mode)
-----------------------------------
1. Build a token-position alignment identifying which positions belong to the
   target entry vs. other entries.
2. Generate on the recipient prompt with every requested component scaled
   in-place (×scale at target positions, ÷scale at other-entry positions)
   in a specified set of layers.
3. Record whether the output label is correct on the recipient prompt.

Sweeps over layer-range configurations are selected with --sweep (nargs="+"):
  isolated            — for each layer l, patch exactly that single layer.
  cumulative          — patch layers [1-1], [1-2], ..., [1-56].
  reverse-cumulative  — patch layers [56-56], [55-56], ..., [1-56].
  all                 — a single run that patches every layer at once.

Raw results are saved as the standard JSON (via base save_results()).
One bar chart per sweep is saved as a PNG.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from typing_extensions import override

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

import mamba_ssm.modules.mamba2 as mamba2_module
from libs.experiments.experiment import Experiment
from libs.models.models import MAMBA_MODELS


NUM_SAMPLES = 500

SWEEP_TITLES = {
    "isolated":            "Isolated (single layer)",
    "cumulative":          "Cumulative [1-k]",
    "reverse-cumulative":  "Reverse cumulative [k-n]",
    "all":                 "All layers",
}

COMPONENT_TITLES = {
    "u":            "Representation (u)",
    "dt_raw":       "dt_raw (pre-bias logit)",
    "delta":        "Δ (post-bias, post-softplus)",
    "B_pre_conv":   "B (pre-conv)",
    "B_post_conv":  "B (post-conv)",
    "C_pre_conv":   "C (pre-conv)",
    "C_post_conv":  "C (post-conv)",
}


# ---------------------------------------------------------------------------
# Alignment dataclass
# ---------------------------------------------------------------------------

@dataclass
class _Alignment:
    """Paired token-position lists for the local and global prompts."""
    # Name + value tokens, entry order (name then value per entry).
    reg_positions: List[int] = field(default_factory=list)  # global prompt positions
    loc_positions: List[int] = field(default_factory=list)  # local prompt positions
    reg_target_positions: List[int] = field(default_factory=list)  # global positions of target entry only
    loc_target_positions: List[int] = field(default_factory=list)  # local positions of target entry only
    # Name-only / value-only / full-entry positions in entry order (for --patching-and-scaling-entry-part).
    reg_name_positions: List[int] = field(default_factory=list)
    loc_name_positions: List[int] = field(default_factory=list)
    reg_number_positions: List[int] = field(default_factory=list)
    loc_number_positions: List[int] = field(default_factory=list)
    # Entire entry: name + colon + value (with leading space) + newline. No flag tokens.
    reg_all_positions: List[int] = field(default_factory=list)
    loc_all_positions: List[int] = field(default_factory=list)
    reg_target_name_positions: List[int] = field(default_factory=list)
    loc_target_name_positions: List[int] = field(default_factory=list)
    reg_target_all_positions: List[int] = field(default_factory=list)
    loc_target_all_positions: List[int] = field(default_factory=list)
    # Number (digit/value) token positions split by target vs. distractor, per prompt side.
    reg_target_number_positions: List[int] = field(default_factory=list)
    reg_other_number_positions: List[int] = field(default_factory=list)
    loc_target_number_positions: List[int] = field(default_factory=list)
    loc_other_number_positions: List[int] = field(default_factory=list)
    regular_prompt: str = ""   # reconstructed flag-free global prompt
    regular_len: int = 0       # token count of the global prompt
    local_prompt: str = ""     # flag-prefixed local prompt
    local_len: int = 0         # token count of the local prompt
    # Only populated for SentencePiece name_start (NVIDIA / Codestral): the
    # local prompt's token ids, assembled from canonical per-segment
    # tokenization rather than by tokenizing `local_prompt` in one shot.
    # name_end leaves this None and feeds the natural
    # string instead. See PatchingAndScaling._assemble_name_start_loc_ids.
    loc_ids: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class PatchingAndScaling(Experiment):
    """Activation-patching sweep over one or more mixer components (dt_raw, u, ...)."""

    # Maps any accepted --dataset value to the "_local" dataset used for sampling.
    # The matching global (flag-free) prompt is reconstructed by _build_alignment().
    _LOCAL_DATASET_BY_BASE = {
        "phonebook": "phonebook_local",
        "phonebook_local": "phonebook_local",
        "profbook": "profbook_local",
        "profbook_local": "profbook_local",
    }
    _GLOBAL_DATASET_BY_LOCAL = {
        "phonebook_local": "phonebook",
        "profbook_local": "profbook",
    }

    def __init__(self, args, model, tokenizer):
        super().__init__(args, model, tokenizer)
        base_dataset = self.args.dataset
        if base_dataset not in self._LOCAL_DATASET_BY_BASE:
            raise ValueError(
                f"{type(self).__name__} only supports --dataset in "
                f"{sorted(self._LOCAL_DATASET_BY_BASE)}, got {base_dataset!r}"
            )
        # Force the settings required for this experiment regardless of CLI args.
        self.args.dataset = self._LOCAL_DATASET_BY_BASE[base_dataset]
        self.args.eval_batch_size = 1
        # Caches for the canonical per-segment tokenization used to assemble
        # local prompt ids on SentencePiece name_start (see _segment_token_ids
        # and _build_alignment). Unused for GPT-NeoX, which tokenizes the
        # natural local string in one shot.
        self._flag_ids_cache: Dict[str, List[int]] = {}
        self._colon_ids_cache: List[int] = None
        self._newline_ids_cache: List[int] = None
        self._dataset_family = self.args.dataset
        self._global_dataset = self._GLOBAL_DATASET_BY_LOCAL[self._dataset_family]
        self._components: List[str] = list(self.args.patching_and_scaling_components)
        self._sweep_types: List[str] = list(self.args.sweep)
        self._patch_donor: str = self.args.patch_donor
        self._patch_recipient: str = self.args.patch_recipient
        self._patch_scale: float = self.args.patch_scale
        self._patching_and_scaling_entries: str = self.args.patching_and_scaling_entries
        self._patching_and_scaling_entry_part: str = self.args.patching_and_scaling_entry_part
        self._is_scaling: bool = self._patch_donor == "scaling"
        if self._patching_and_scaling_entries not in ("all_entries", "target_entry", "non_target_entries"):
            raise ValueError(
                f"--patching-and-scaling-entries must be one of "
                f"{{'all_entries', 'target_entry', 'non_target_entries'}}, "
                f"got {self._patching_and_scaling_entries!r}"
            )
        if self._patching_and_scaling_entry_part not in ("name", "number", "both", "all"):
            raise ValueError(
                f"--patching-and-scaling-entry-part must be one of {{'name', 'number', 'both', 'all'}}, "
                f"got {self._patching_and_scaling_entry_part!r}"
            )
        if self._is_scaling:
            if self._patch_recipient not in ("local", "global"):
                raise ValueError(f"--patch-recipient must be 'local' or 'global', got {self._patch_recipient!r}")
        else:
            for name, value in (("patch_donor", self._patch_donor), ("patch_recipient", self._patch_recipient)):
                if value not in ("local", "global"):
                    raise ValueError(f"--{name.replace('_', '-')} must be 'local' or 'global', got {value!r}")
            if self._patch_donor == self._patch_recipient:
                raise ValueError(
                    f"--patch-donor and --patch-recipient must differ, "
                    f"got both {self._patch_donor!r}"
                )
        # Tag identifying which component set is jointly patched in this run, used to
        # namespace output files so different runs don't overwrite each other's results.
        self._component_tag = "+".join(self._components)
        if self._is_scaling:
            scale_str = str(self._patch_scale).rstrip("0").rstrip(".")
            self._direction_tag = f"{self._patch_recipient}{scale_str}x"
        else:
            self._direction_tag = f"{self._patch_donor}_to_{self._patch_recipient}"

        self.patching_tokens_tag = f"patch-{self._patching_and_scaling_entries}_{self._patching_and_scaling_entry_part}-prompt-{self.args.local_dataset_flag_entries}"

    @property
    @override
    def _experiment_subdir(self) -> str:
        """Relative path segment shared by both output roots: '{model}_{task}/{experiment_name}'."""
        return os.path.join(
            f"{self.args.model}_{self.args.task}",
            self._experiment_name
        )
    
    @property
    @override
    def _output_path_extra(self) -> tuple[str, ...]:
        return (self._component_tag, self._direction_tag)

    @property
    def _variant_prefix(self):
        return f"{self.patching_tokens_tag}-"

    def _dataset_for_side(self, side: str) -> str:
        return self._dataset_family if side == "local" else self._global_dataset

    @property
    def _donor_dataset(self) -> str:
        if self._is_scaling:
            return None
        return self._dataset_for_side(self._patch_donor)

    @property
    def _recipient_dataset(self) -> str:
        return self._dataset_for_side(self._patch_recipient)

    @property
    def _patch_direction_label(self) -> str:
        if self._is_scaling:
            return f"scale x{self._patch_scale} on {self._recipient_dataset}"
        return f"{self._donor_dataset} → {self._recipient_dataset}"

    # ------------------------------------------------------------------
    # Sweep configuration builders
    # ------------------------------------------------------------------

    @staticmethod
    def _sweep_isolated(n_layer: int) -> List[Tuple[str, Set[int]]]:
        """One config per layer: patch exactly that single layer (1-indexed label)."""
        return [(str(l), {l - 1}) for l in range(1, n_layer + 1)]

    @staticmethod
    def _sweep_cumulative(n_layer: int) -> List[Tuple[str, Set[int]]]:
        """Cumulative [1-k] for k in 1..n_layer."""
        return [(f"1-{k}", set(range(0, k))) for k in range(1, n_layer + 1)]

    @staticmethod
    def _sweep_reverse_cumulative(n_layer: int) -> List[Tuple[str, Set[int]]]:
        """Reverse cumulative [k-n_layer] for k in n_layer..1."""
        return [
            (f"{k}-{n_layer}", set(range(k - 1, n_layer)))
            for k in range(n_layer, 0, -1)
        ]

    @staticmethod
    def _sweep_all(n_layer: int) -> List[Tuple[str, Set[int]]]:
        """A single config: patch every layer at once."""
        return [(f"1-{n_layer}", set(range(n_layer)))]

    _SWEEP_BUILDERS = {
        "isolated": _sweep_isolated,
        "cumulative": _sweep_cumulative,
        "reverse-cumulative": _sweep_reverse_cumulative,
        "all": _sweep_all,
    }

    # ------------------------------------------------------------------
    # Token-position alignment
    # ------------------------------------------------------------------

    def _tok_len(self, text: str) -> int:
        """Number of tokens in `text` when appended to an empty context."""
        enc = self.tokenizer(text)
        ids = enc.input_ids if hasattr(enc, "input_ids") else enc
        return len(ids)

    def _tok_len_incremental(self, accumulated: str, text: str) -> int:
        """Number of NEW tokens added by appending `text` to `accumulated`."""
        before = self._tok_len(accumulated)
        after = self._tok_len(accumulated + text)
        return after - before

    def _tok_ids(self, text: str) -> List[int]:
        """Token ids for `text` tokenized on its own (includes any BOS token)."""
        enc = self.tokenizer(text)
        return enc.input_ids if hasattr(enc, "input_ids") else list(enc)

    def _segment_token_ids(self, context: str, text: str) -> List[int]:
        """Token ids for `text` as it tokenizes right after a fixed `context`,
        with `context`'s own tokens (including BOS) stripped off.

        Used to build a *canonical* tokenization of a name/flag/value/separator
        segment that does not depend on whatever text happens to precede it in
        the final rendered prompt — e.g. tokenizing "\\n" + name and dropping the
        "\\n" tokens gives the name's tokens as they appear after a newline,
        regardless of what actually precedes that newline. Not cached here since
        `text` (name/value) varies per entry; callers cache by `context` alone
        where `text` is also fixed (flag words, separators).
        """
        context_ids = self._tok_ids(context)
        return self._tok_ids(context + text)[len(context_ids):]

    def _name_token_ids(self, name: str) -> List[int]:
        """Canonical token ids for an entry name, as it tokenizes right after
        a newline (the context every entry name — flagged or not — appears
        in on the regular/global side, and that unflagged local entries also
        share). Used for both prompt sides so name tokens match exactly.
        """
        return self._segment_token_ids("\n", name)

    def _flag_token_ids(self, flag_word: str) -> List[int]:
        """Canonical token ids for `flag_word + " "` (e.g. "target " / "skip
        "), as it tokenizes right after a newline. Cached per flag word since
        there are only two possible values and they recur constantly.
        """
        cache = self._flag_ids_cache
        if flag_word not in cache:
            cache[flag_word] = self._segment_token_ids("\n", flag_word + " ")
        return cache[flag_word]

    def _value_with_leading_space_ids(self, value: str) -> List[int]:
        """Canonical token ids for " " + value (a leading space followed by
        the entry's value — phone digits or a profession word) as it
        tokenizes right after a bare ":" — the real context it always
        appears in on both prompt sides (straight after the colon token,
        before any value characters).

        The leading space cannot be split off as its own separate, reusable
        token the way ": " (colon-space) initially seemed to suggest: for
        word values (profbook), SentencePiece always fuses the space into the
        value's first token (e.g. "▁psychologist" as one piece) — there is no
        clean boundary between "the separator's space" and "the value" at
        all for these; digits happen not to fuse this way, but treating the
        space as an independent, pre-existing token there too would silently
        double-count it once actually appended to the colon. Bundling the
        space with the value (and never counting it separately) sidesteps
        this ambiguity entirely and simply matches whatever the real,
        single-pass tokenization of ": "+value actually produces.
        """
        return self._segment_token_ids(":", " " + value)

    @property
    def _colon_ids(self) -> List[int]:
        """Canonical token ids for the bare ":" ending an entry name. Verified
        stable (a single, dedicated token) regardless of what precedes it —
        the end of a name or a bare newline — so a fixed cached value (via
        the "\\n" context) is used.
        """
        if self._colon_ids_cache is None:
            self._colon_ids_cache = self._segment_token_ids("\n", ":")
        return self._colon_ids_cache

    @property
    def _newline_ids(self) -> List[int]:
        """Canonical token ids for the "\\n" separating entries. Verified
        stable regardless of what value (digits or words) precedes it, so a
        fixed cached value (via an arbitrary digit context) is used.
        """
        if self._newline_ids_cache is None:
            self._newline_ids_cache = self._segment_token_ids("0", "\n")
        return self._newline_ids_cache

    @property
    def _assemble_name_start_loc_ids(self) -> bool:
        """Whether name_start local prompts must be assembled from canonical
        segments rather than tokenized as a full string.

        SentencePiece (NVIDIA LlamaTokenizer, Codestral) can merge the flag
        word into the following name, so a 1:1 local<->global name alignment
        is only recoverable by assembling loc_ids.
        """
        return "GPTNeoX" not in type(self.tokenizer).__name__

    def _build_alignment(self, prompt_parts: dict, local_prompt: str, target_entry_index: int = -1) -> _Alignment:
        """Build paired token-position lists for name/value/separator tokens.

        Walks the local and global prompts in parallel, segment-by-segment.
        For each entry (phonebook or profbook) the flag word is stripped to
        produce the corresponding global-prompt segment; token counts for
        name, colon, value, and newline spans are asserted to match between
        both prompts.

        For --local-dataset-flag-location=name_end, the local side is the
        natural string tokenized once, with character spans mapped to token
        ranges (names always follow "\\n" and values always follow a space,
        on both prompt sides).

        For --local-dataset-flag-location=name_start the local path depends
        on the tokenizer (see _assemble_name_start_loc_ids):

        - SentencePiece (NVIDIA / Codestral): the flag word's trailing space
          merges into the following name (or, for some names, fuses the flag
          word and the start of the name into a single unsplittable token),
          so a clean local<->global name split is not recoverable from the
          natural string. Local token ids are assembled from independently,
          canonically tokenized segments (_name_token_ids / _flag_token_ids /
          _value_with_leading_space_ids / _colon_ids / _newline_ids). That
          makes the flagged entry's name tokens byte-identical to the
          regular side (both computed after a newline), at the cost of a
          small deviation from tokenizing the rendered string in one pass.

        Returns an _Alignment with:
          - reg_positions / loc_positions : parallel lists of absolute token
            indices (0-based, relative to the start of the prompt) for name
            and value tokens interleaved per entry.
          - reg_name_positions / loc_name_positions : name tokens only, in
            entry order (used when --patching-and-scaling-entry-part=name).
          - reg_number_positions / loc_number_positions : value tokens only,
            in entry order (used when --patching-and-scaling-entry-part=number).
          - reg_all_positions / loc_all_positions : entire entry in order
            (name, colon, value with leading space, newline). Flag words are
            excluded (used when --patching-and-scaling-entry-part=all).
          - reg_target_positions / loc_target_positions : same as
            reg/loc_positions, but only for the target entry
            (entry_idx == target_entry_index).
          - reg_target_name_positions / loc_target_name_positions : target
            entry name tokens only.
          - reg_target_all_positions / loc_target_all_positions : target
            entry's full-entry tokens (same span as reg/loc_all_positions).
          - regular_prompt : the reconstructed flag-free global prompt string.
          - regular_len    : total token count of the global prompt.
          - local_prompt   : the flag-prefixed local prompt string.
          - local_len      : total token count of the local prompt.
          - loc_ids        : assembled local token ids for SentencePiece
            name_start; None for name_end and for GPT-NeoX name_start.
        """
        prefix_local = prompt_parts["prefix"]
        entries = prompt_parts["entries"]
        suffix = prompt_parts["suffix"]
        flag_location = self.args.local_dataset_flag_location
        flag_entries = self.args.local_dataset_flag_entries

        if flag_location not in ("name_end", "name_start"):
            raise ValueError(f"Unknown --local-dataset-flag-location value: {flag_location!r}")

        # Strip the flag word from the example entries embedded in the prefix,
        # regardless of where it was placed:
        #   name_end:   "Name: skip Phone"  → "Name: Phone"
        #   name_start: "skip Name: Phone"  → "Name: Phone"
        prefix_regular = re.sub(r": (?:skip|target) ", ": ", prefix_local)
        prefix_regular = re.sub(r"(^|\n)(?:skip|target) ", r"\1", prefix_regular)

        result = _Alignment()

        # --- Pass 1: build prompt text(s) and record CHARACTER offsets for
        # each entry's name/colon/value/newline span. No tokenization happens
        # in this pass.
        #
        # Rationale: retokenizing a *growing* accumulated string after every
        # segment (the old approach) is unreliable for values too, not just
        # names — e.g. "Allison Hill: " tokenizes with a trailing solo space
        # token, but "Allison Hill: psychologist" tokenizes that same space
        # fused into a single "▁psychologist" token, so re-tokenizing before
        # vs. after appending the value can report *zero* new tokens even
        # though "psychologist" plainly needs to be represented somehow. The
        # only fully reliable way to find a span's real token boundaries is
        # to tokenize the complete, final string exactly once (pass 2) and
        # locate boundaries via decoded-prefix-length binary search.
        reg_text = prefix_regular
        # Each entry: (is_target, name_span, colon_span, val_span, newline_span)
        # as (start_char, end_char) pairs. Flag tokens (name_end local) sit
        # between colon and value and are deliberately omitted.
        reg_entries: List[Tuple[bool, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []

        use_assembled_loc_ids = (
            flag_location == "name_start" and self._assemble_name_start_loc_ids
        )

        if use_assembled_loc_ids:
            loc_ids: List[int] = self._tok_ids(prefix_local)
            # Per-entry token ranges, assembled directly (no char offsets needed).
            loc_name_ranges: List[List[int]] = []
            loc_colon_ranges: List[List[int]] = []
            loc_val_ranges: List[List[int]] = []
            loc_newline_ranges: List[List[int]] = []

            def collect_loc_ids(segment_ids: List[int]) -> List[int]:
                start = len(loc_ids)
                loc_ids.extend(segment_ids)
                return list(range(start, len(loc_ids)))
        else:
            # name_end, or GPT-NeoX name_start: natural string + char spans.
            loc_text = prefix_local
            loc_entries: List[Tuple[bool, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []

        for entry_idx, entry_str in enumerate(entries):
            # entry_str has a trailing "\n"; strip it for parsing.
            entry_body = entry_str.rstrip("\n")
            is_target = entry_idx == target_entry_index

            # Mirrors PhonebookEntry.__str__ / ProfbookEntry.__str__: whether (and
            # which) flag word is present on this entry depends on --local-dataset-
            # -flag-entries. With "target" (the common case) only the target entry
            # is flagged; all other entries render as plain "Name: Value".
            flag_word = ""
            if is_target and flag_entries in ("target", "target_and_skip"):
                flag_word = "target"
            elif not is_target and flag_entries in ("skip", "target_and_skip"):
                flag_word = "skip"

            if flag_location == "name_end":
                # local (flagged):   "First Last: skip 1234567890"
                # local (unflagged): "First Last: 1234567890"
                # regular:           "First Last: 1234567890" / "First Last: engineer"
                entry_name, rest = entry_body.split(": ", 1)
                if flag_word:
                    parsed_flag_word, entry_value = rest.split(" ", 1)
                    assert parsed_flag_word == flag_word, (
                        f"Entry {entry_idx}: expected flag {flag_word!r}, got {parsed_flag_word!r}"
                    )
                else:
                    entry_value = rest

                # --- Local: append name, ":", [space,flag_word]?, space, value,
                # "\n"; record offsets. The value span's start is placed right
                # BEFORE the space that immediately precedes the value (after
                # "flag_word" when flagged, after ":" otherwise) rather than
                # after it: SentencePiece fuses that space into the value's
                # first token for word values (e.g. profession names), so
                # there is no clean token boundary right after the space —
                # only right before it. Digit values happen not to fuse this
                # way, but including the space in the "value" span either way
                # is harmless (it isn't itself patched at a separate position).
                loc_name_start = len(loc_text)
                loc_text += entry_name
                loc_colon_start = len(loc_text)
                loc_text += ":"
                loc_colon_end = len(loc_text)
                if flag_word:
                    loc_text += " " + flag_word
                loc_val_start = len(loc_text)
                loc_text += " " + entry_value
                loc_val_end = len(loc_text)
                loc_nl_start = len(loc_text)
                loc_text += "\n"
                loc_entries.append(
                    (
                        is_target,
                        (loc_name_start, loc_name_start + len(entry_name)),
                        (loc_colon_start, loc_colon_end),
                        (loc_val_start, loc_val_end),
                        (loc_nl_start, len(loc_text)),
                    )
                )
            else:  # name_start
                # local (flagged):   "skip First Last: 1234567890"
                # local (unflagged): "First Last: 1234567890"
                # regular:           "First Last: 1234567890" / "First Last: engineer"
                if flag_word:
                    flag_prefix = flag_word + " "
                    assert entry_body.startswith(flag_prefix), (
                        f"Entry {entry_idx}: expected prefix {flag_prefix!r} in {entry_body!r}"
                    )
                    rest = entry_body[len(flag_prefix):]
                else:
                    rest = entry_body
                entry_name, entry_value = rest.split(": ", 1)

                if use_assembled_loc_ids:
                    # SentencePiece: assemble ids for [flag?] + name + ":" +
                    # " "+value + "\n". The value span includes its leading
                    # space (see _value_with_leading_space_ids) rather than
                    # treating ": " as one fixed 2-token separator, for the
                    # same reason as the name_end branch above: that space
                    # fuses into word values' first token.
                    if flag_word:
                        collect_loc_ids(self._flag_token_ids(flag_word))
                    loc_name_ranges.append(collect_loc_ids(self._name_token_ids(entry_name)))
                    loc_colon_ranges.append(collect_loc_ids(self._colon_ids))
                    loc_val_ranges.append(collect_loc_ids(self._value_with_leading_space_ids(entry_value)))
                    loc_newline_ranges.append(collect_loc_ids(self._newline_ids))
                else:
                    # GPT-NeoX: record char offsets on the natural string.
                    # Flag word is rest. Its trailing space is bundled with
                    # the name (BPE fuses it into ĠName); the only clean
                    # token boundary is immediately before that space.
                    if flag_word:
                        loc_text += flag_word
                        loc_name_start = len(loc_text)
                        loc_text += " " + entry_name
                    else:
                        loc_name_start = len(loc_text)
                        loc_text += entry_name
                    loc_colon_start = len(loc_text)
                    loc_text += ":"
                    loc_colon_end = len(loc_text)
                    loc_val_start = len(loc_text)
                    loc_text += " " + entry_value
                    loc_val_end = len(loc_text)
                    loc_nl_start = len(loc_text)
                    loc_text += "\n"
                    loc_entries.append(
                        (
                            is_target,
                            (loc_name_start, loc_colon_start),
                            (loc_colon_start, loc_colon_end),
                            (loc_val_start, loc_val_end),
                            (loc_nl_start, len(loc_text)),
                        )
                    )

            # --- Regular: append name, ":", " "+value, "\n"; record offsets.
            # Value span starts right before its leading space (see comment
            # in the name_end branch above for why).
            reg_name_start = len(reg_text)
            reg_text += entry_name
            reg_colon_start = len(reg_text)
            reg_text += ":"
            reg_colon_end = len(reg_text)
            reg_val_start = len(reg_text)
            reg_text += " " + entry_value
            reg_val_end = len(reg_text)
            reg_nl_start = len(reg_text)
            reg_text += "\n"
            reg_entries.append(
                (
                    is_target,
                    (reg_name_start, reg_name_start + len(entry_name)),
                    (reg_colon_start, reg_colon_end),
                    (reg_val_start, reg_val_end),
                    (reg_nl_start, len(reg_text)),
                )
            )

        reg_text += suffix

        # --- Pass 2: tokenize the regular side's complete text exactly once,
        # then map every recorded character offset to its real token index
        # via decoded-prefix-length binary search (robust to any tokenization
        # instability from adjacent text, since it reads boundaries directly
        # off the actual final tokenization rather than assuming any).
        reg_ids_full = self._tok_ids(reg_text)
        (
            reg_name_ranges, reg_colon_ranges, reg_val_ranges, reg_newline_ranges,
        ) = self._char_spans_to_token_ranges(
            reg_ids_full,
            [
                (name_span, colon_span, val_span, nl_span)
                for _, name_span, colon_span, val_span, nl_span in reg_entries
            ],
        )
        result.regular_prompt = reg_text
        result.regular_len = len(reg_ids_full)

        if use_assembled_loc_ids:
            # Suffix always follows the last entry's "\n" — same canonical
            # "after a newline" context used throughout.
            loc_ids.extend(self._segment_token_ids("\n", suffix))
            result.local_prompt = local_prompt
            result.local_len = len(loc_ids)
            result.loc_ids = loc_ids
            local_is_target = [e[0] for e in reg_entries]
        else:
            loc_text += suffix
            if loc_text != local_prompt:
                raise AssertionError(
                    "Reconstructed local prompt does not match the dataset string; "
                    "token positions would not line up with the fed prompt."
                )
            loc_ids_full = self._tok_ids(loc_text)
            (
                loc_name_ranges, loc_colon_ranges, loc_val_ranges, loc_newline_ranges,
            ) = self._char_spans_to_token_ranges(
                loc_ids_full,
                [
                    (name_span, colon_span, val_span, nl_span)
                    for _, name_span, colon_span, val_span, nl_span in loc_entries
                ],
            )
            result.local_prompt = local_prompt
            result.local_len = len(loc_ids_full)
            result.loc_ids = None
            local_is_target = [e[0] for e in loc_entries]

        # Number / colon / newline token counts must match entry-by-entry:
        # they share the same context (": " / "\n") on both prompts. Name
        # counts match for name_end and for SentencePiece name_start (the
        # latter by construction). On GPT-NeoX name_start they often do
        # *not*: a flagged name after "target " is ĠJennifer (one piece)
        # while the same name after a newline on the global side splits
        # (J + ennifer). Skipping those samples drops most of the dataset.
        # Right-align to the shared suffix (typically ĠLast, plus any
        # first-name pieces that happen to match in count) and keep the
        # natural local string. Unflagged GPT-NeoX names still follow a
        # newline on both sides, so a mismatch there is a real bug.
        allow_flagged_name_realign = (
            flag_location == "name_start" and not use_assembled_loc_ids
        )
        for entry_idx, (name_range, reg_name_range) in enumerate(zip(loc_name_ranges, reg_name_ranges)):
            if len(name_range) == len(reg_name_range):
                continue
            is_flagged = (
                (local_is_target[entry_idx] and flag_entries in ("target", "target_and_skip"))
                or ((not local_is_target[entry_idx]) and flag_entries in ("skip", "target_and_skip"))
            )
            if allow_flagged_name_realign and is_flagged:
                n = min(len(name_range), len(reg_name_range))
                loc_name_ranges[entry_idx] = name_range[-n:]
                reg_name_ranges[entry_idx] = reg_name_range[-n:]
                continue
            name_start, name_end = reg_entries[entry_idx][1]
            entry_name = reg_text[name_start:name_end]
            raise AssertionError(
                f"Entry {entry_idx} ('{entry_name}'): name tokenizes to "
                f"{len(name_range)} tokens in local but {len(reg_name_range)} in regular."
            )
        for entry_idx, (val_range, reg_val_range) in enumerate(zip(loc_val_ranges, reg_val_ranges)):
            if len(val_range) != len(reg_val_range):
                raise AssertionError(
                    f"Entry {entry_idx}: value tokenizes to {len(val_range)} tokens in "
                    f"local but {len(reg_val_range)} in regular."
                )
        for entry_idx, (colon_range, reg_colon_range) in enumerate(zip(loc_colon_ranges, reg_colon_ranges)):
            if len(colon_range) != len(reg_colon_range):
                raise AssertionError(
                    f"Entry {entry_idx}: colon tokenizes to {len(colon_range)} tokens in "
                    f"local but {len(reg_colon_range)} in regular."
                )
        for entry_idx, (newline_range, reg_newline_range) in enumerate(zip(loc_newline_ranges, reg_newline_ranges)):
            if len(newline_range) != len(reg_newline_range):
                raise AssertionError(
                    f"Entry {entry_idx}: newline tokenizes to {len(newline_range)} tokens in "
                    f"local but {len(reg_newline_range)} in regular."
                )

        for (is_target, *_), name_range, colon_range, val_range, newline_range in zip(
            reg_entries, reg_name_ranges, reg_colon_ranges, reg_val_ranges, reg_newline_ranges,
        ):
            all_range = name_range + colon_range + val_range + newline_range
            result.reg_positions.extend(name_range)
            result.reg_positions.extend(val_range)
            result.reg_name_positions.extend(name_range)
            result.reg_number_positions.extend(val_range)
            result.reg_all_positions.extend(all_range)
            if is_target:
                result.reg_target_positions.extend(name_range)
                result.reg_target_positions.extend(val_range)
                result.reg_target_name_positions.extend(name_range)
                result.reg_target_number_positions.extend(val_range)
                result.reg_target_all_positions.extend(all_range)
            else:
                result.reg_other_number_positions.extend(val_range)

        for is_target, name_range, colon_range, val_range, newline_range in zip(
            local_is_target, loc_name_ranges, loc_colon_ranges, loc_val_ranges, loc_newline_ranges,
        ):
            all_range = name_range + colon_range + val_range + newline_range
            result.loc_positions.extend(name_range)
            result.loc_positions.extend(val_range)
            result.loc_name_positions.extend(name_range)
            result.loc_number_positions.extend(val_range)
            result.loc_all_positions.extend(all_range)
            if is_target:
                result.loc_target_positions.extend(name_range)
                result.loc_target_positions.extend(val_range)
                result.loc_target_name_positions.extend(name_range)
                result.loc_target_number_positions.extend(val_range)
                result.loc_target_all_positions.extend(all_range)
            else:
                result.loc_other_number_positions.extend(val_range)

        return result

    def _char_spans_to_token_ranges(
        self, ids: List[int], spans: List[Tuple[Tuple[int, int], ...]]
    ) -> Tuple[List[List[int]], ...]:
        """Map each entry's character-offset spans to token-index ranges
        within `ids`, the single, final tokenization of the complete text
        these offsets were recorded against.

        `spans` is a list of entries, each a tuple of (start, end) char
        spans (e.g. name/colon/value/newline). Returns one list-of-ranges
        per span slot, in the same order.

        Uses binary search over decoded-prefix length (monotonic in token
        count for a fixed `ids`, and exact — verified: decoding growing
        prefixes of one fixed tokenization reconstructs exact text prefixes,
        even though re-tokenizing different substrings independently can
        disagree on token boundaries).
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

        n_slots = len(spans[0]) if spans else 0
        slot_ranges: List[List[List[int]]] = [[] for _ in range(n_slots)]
        for entry_spans in spans:
            for slot, (start, end) in enumerate(entry_spans):
                slot_ranges[slot].append(list(range(char_to_tok(start), char_to_tok(end))))
        return tuple(slot_ranges)

    def _prompt_for_side(self, alignment: _Alignment, side: str) -> "str | List[int]":
        """Return whatever _generate_with_hooks should feed to the model for
        `side`: pre-built token ids when available (SentencePiece name_start
        local), otherwise the prompt text (tokenized normally).
        """
        if side == "local":
            return alignment.loc_ids if alignment.loc_ids is not None else alignment.local_prompt
        return alignment.regular_prompt

    def _positions_for_side(self, alignment: _Alignment, side: str) -> List[int]:
        """All-entry token positions on `side`, filtered by --patching-and-scaling-entry-part."""
        part = self._patching_and_scaling_entry_part
        if side == "local":
            by_part = {
                "name": alignment.loc_name_positions,
                "number": alignment.loc_number_positions,
                "both": alignment.loc_positions,
                "all": alignment.loc_all_positions,
            }
        else:
            by_part = {
                "name": alignment.reg_name_positions,
                "number": alignment.reg_number_positions,
                "both": alignment.reg_positions,
                "all": alignment.reg_all_positions,
            }
        return by_part[part]

    def _target_positions_for_side(self, alignment: _Alignment, side: str) -> List[int]:
        """Target-entry token positions on `side`, filtered by --patching-and-scaling-entry-part."""
        part = self._patching_and_scaling_entry_part
        if side == "local":
            by_part = {
                "name": alignment.loc_target_name_positions,
                "number": alignment.loc_target_number_positions,
                "both": alignment.loc_target_positions,
                "all": alignment.loc_target_all_positions,
            }
        else:
            by_part = {
                "name": alignment.reg_target_name_positions,
                "number": alignment.reg_target_number_positions,
                "both": alignment.reg_target_positions,
                "all": alignment.reg_target_all_positions,
            }
        return by_part[part]

    def _entry_positions_for_side(self, alignment: _Alignment, side: str) -> List[int]:
        """Entry token positions on `side`, filtered by --patching-and-scaling-entries
        and --patching-and-scaling-entry-part.
        """
        all_positions = self._positions_for_side(alignment, side)
        if self._patching_and_scaling_entries == "all_entries":
            return all_positions
        target_positions = self._target_positions_for_side(alignment, side)
        if self._patching_and_scaling_entries == "target_entry":
            return target_positions
        if self._patching_and_scaling_entries == "non_target_entries":
            target_set = set(target_positions)
            return [p for p in all_positions if p not in target_set]
        raise ValueError(f"Unknown --patching-and-scaling-entries value: {self._patching_and_scaling_entries!r}")

    def _len_for_side(self, alignment: _Alignment, side: str) -> int:
        return alignment.local_len if side == "local" else alignment.regular_len

    def _prompts_from_alignment(self, alignment: _Alignment) -> Tuple[str, str]:
        """Return (donor_prompt, recipient_prompt) for the current patch direction."""
        return (
            self._prompt_for_side(alignment, self._patch_donor),
            self._prompt_for_side(alignment, self._patch_recipient),
        )

    def _configure_hook_contexts(self, hook_contexts: Dict[str, dict], alignment: _Alignment) -> None:
        """Set donor/recipient token positions and recipient length on every hook context."""
        donor_positions = self._entry_positions_for_side(alignment, self._patch_donor)
        recipient_positions = self._entry_positions_for_side(alignment, self._patch_recipient)
        recipient_len = self._len_for_side(alignment, self._patch_recipient)

        for hook_context in hook_contexts.values():
            hook_context["reg_positions"] = recipient_positions
            hook_context["loc_positions"] = donor_positions
            hook_context["regular_len"] = recipient_len
            hook_context["donor"] = {}

    def _configure_hook_contexts_scaling(self, hook_contexts: Dict[str, dict], alignment: _Alignment) -> None:
        """Configure hook contexts for scaling mode (no donor; scale in-place on recipient)."""
        positions = self._entry_positions_for_side(alignment, self._patch_recipient)
        target_positions = set(self._target_positions_for_side(alignment, self._patch_recipient))
        length = self._len_for_side(alignment, self._patch_recipient)

        for hook_context in hook_contexts.values():
            hook_context["reg_positions"] = positions
            hook_context["loc_positions"] = positions
            hook_context["regular_len"] = length
            hook_context["target_positions"] = target_positions
            hook_context["patch_scale"] = self._patch_scale
            hook_context["donor"] = {}

    def _scale_factor_tensor(self, hook_context: dict, positions: List[int], device) -> torch.Tensor:
        """Return a per-position scale factor tensor (scale for target, 1/scale for others)."""
        target_set = hook_context.get("target_positions", set())
        scale = hook_context.get("patch_scale", 2.0)
        inv_scale = 1.0 / scale
        factors = [scale if p in target_set else inv_scale for p in positions]
        return torch.tensor(factors, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Hook registration — one implementation per component
    # ------------------------------------------------------------------

    def _register_hooks_dt_raw(self, n_layer: int, n_heads: int, hook_context: dict) -> List:
        """Register one forward hook per layer on mixer.in_proj, acting on dt_raw.

        The hook behaviour is controlled by `hook_context`:
          mode         : 'capture' | 'patch' | 'off'
          active_layers: set of 0-indexed layer indices to act on (patch only)
          donor        : dict layer_idx -> (T_local, n_heads) tensor (patch only)
          reg_positions: list of int — recipient token indices to overwrite
          loc_positions: list of int — donor token indices to read from
          regular_len  : int — expected sequence length of the prefill to guard
                         against decode-step calls
        """
        hooks = []

        for l in range(n_layer):
            def make_hook(layer_idx: int):
                def hook_fn(module, input, output):
                    mode = hook_context.get("mode", "off")

                    if mode == "capture":
                        # output shape during prefill: (1, T, d_in_proj)
                        if output.dim() == 3 and output.shape[1] > 1:
                            # Store the dt_raw slice: last n_heads columns.
                            dt_raw = output[0, :, -n_heads:].detach().clone()  # (T, n_heads)
                            hook_context["donor"][layer_idx] = dt_raw
                        # No return — output unchanged.

                    elif mode == "patch":
                        active = hook_context.get("active_layers", set())
                        if layer_idx not in active:
                            return  # Not in patching range; leave untouched.

                        # Only act on the prefill, not per-token decode steps.
                        if output.dim() != 3 or output.shape[1] != hook_context.get("regular_len", -1):
                            return

                        donor = hook_context.get("donor", {})
                        if layer_idx not in donor:
                            return

                        reg_pos = hook_context.get("reg_positions", [])
                        loc_pos = hook_context.get("loc_positions", [])
                        if not reg_pos:
                            return

                        patched = output.clone()
                        donor_dt = donor[layer_idx]  # (T_local, n_heads)
                        reg_idx = torch.tensor(reg_pos, device=output.device, dtype=torch.long)
                        loc_idx = torch.tensor(loc_pos, device=output.device, dtype=torch.long)
                        # Overwrite dt_raw columns (last n_heads) at aligned positions.
                        patched[0, reg_idx, -n_heads:] = donor_dt[loc_idx].to(patched.dtype)
                        return patched

                    elif mode == "scale":
                        active = hook_context.get("active_layers", set())
                        if layer_idx not in active:
                            return

                        if output.dim() != 3 or output.shape[1] != hook_context.get("regular_len", -1):
                            return

                        reg_pos = hook_context.get("reg_positions", [])
                        if not reg_pos:
                            return

                        reg_idx = torch.tensor(reg_pos, device=output.device, dtype=torch.long)
                        factors = self._scale_factor_tensor(hook_context, reg_pos, output.device)
                        patched = output.clone()
                        # factors shape: (len(reg_pos),); broadcast over n_heads dim.
                        patched[0, reg_idx, -n_heads:] = (
                            output[0, reg_idx, -n_heads:].float() * factors.unsqueeze(1)
                        ).to(patched.dtype)
                        return patched

                    # mode == 'off': do nothing.

                return hook_fn

            mixer = self.model.backbone.layers[l].mixer
            h = mixer.in_proj.register_forward_hook(make_hook(l))
            hooks.append(h)

        return hooks

    def _register_hooks_u(self, n_layer: int, hook_context: dict) -> List:
        """Register one forward pre-hook per layer on mixer.in_proj, acting on u.

        The hook behaviour is controlled by `hook_context`:
          mode         : 'capture' | 'patch' | 'off'
          active_layers: set of 0-indexed layer indices to act on (patch only)
          donor        : dict layer_idx -> (T_local, d_model) tensor (patch only)
          reg_positions: list of int — recipient token indices to overwrite
          loc_positions: list of int — donor token indices to read from
          regular_len  : int — expected sequence length of the prefill to guard
                         against decode-step calls

        In capture mode, stores u (in_proj input) for each layer.
        In patch mode, overwrites the full d_model representation at aligned positions.
        """
        hooks = []

        for l in range(n_layer):
            def make_hook(layer_idx: int):
                def hook_fn(module, args):
                    mode = hook_context.get("mode", "off")
                    x = args[0]

                    if mode == "capture":
                        # x shape during prefill: (1, T, d_model)
                        if x.dim() == 3 and x.shape[1] > 1:
                            hook_context["donor"][layer_idx] = x[0].detach().clone()  # (T, d_model)
                        return None

                    elif mode == "patch":
                        active = hook_context.get("active_layers", set())
                        if layer_idx not in active:
                            return None

                        # Only act on the prefill, not per-token decode steps.
                        if x.dim() != 3 or x.shape[1] != hook_context.get("regular_len", -1):
                            return None

                        donor = hook_context.get("donor", {})
                        if layer_idx not in donor:
                            return None

                        reg_pos = hook_context.get("reg_positions", [])
                        loc_pos = hook_context.get("loc_positions", [])
                        if not reg_pos:
                            return None

                        patched = x.clone()
                        donor_x = donor[layer_idx]  # (T_local, d_model)
                        reg_idx = torch.tensor(reg_pos, device=x.device, dtype=torch.long)
                        loc_idx = torch.tensor(loc_pos, device=x.device, dtype=torch.long)
                        # Overwrite full representation at aligned positions.
                        patched[0, reg_idx, :] = donor_x[loc_idx].to(patched.dtype)
                        return (patched,) + args[1:]

                    elif mode == "scale":
                        active = hook_context.get("active_layers", set())
                        if layer_idx not in active:
                            return None

                        if x.dim() != 3 or x.shape[1] != hook_context.get("regular_len", -1):
                            return None

                        reg_pos = hook_context.get("reg_positions", [])
                        if not reg_pos:
                            return None

                        reg_idx = torch.tensor(reg_pos, device=x.device, dtype=torch.long)
                        factors = self._scale_factor_tensor(hook_context, reg_pos, x.device)
                        patched = x.clone()
                        # factors shape: (len(reg_pos),); broadcast over d_model dim.
                        patched[0, reg_idx, :] = (
                            x[0, reg_idx, :].float() * factors.unsqueeze(1)
                        ).to(patched.dtype)
                        return (patched,) + args[1:]

                    # mode == 'off': do nothing.
                    return None

                return hook_fn

            mixer = self.model.backbone.layers[l].mixer
            h = mixer.in_proj.register_forward_pre_hook(make_hook(l))
            hooks.append(h)

        return hooks

    @staticmethod
    def _in_proj_slice_bounds(mixer, part: str) -> Tuple[int, int]:
        """Column bounds of `part` within a Mamba2 mixer.in_proj output.

        Layout (see mamba_ssm.modules.mamba2.Mamba2.forward): the in_proj output is
        [z0, x0, z, xBC, dt] where xBC further splits into [x, B, C]. z0/x0 (the
        gated-MLP branch) have width d_mlp, which is zero unless d_ssm < d_inner.
        """
        d_ssm = mixer.d_ssm
        ngroups = mixer.ngroups
        d_state = mixer.d_state
        n_heads = mixer.nheads
        d_in_proj = mixer.in_proj.out_features
        d_mlp = (d_in_proj - 2 * d_ssm - 2 * ngroups * d_state - n_heads) // 2

        xBC_start = 2 * d_mlp + d_ssm
        B_start = xBC_start + d_ssm
        C_start = B_start + ngroups * d_state
        dt_start = d_in_proj - n_heads

        bounds = {
            "B_pre_conv": (B_start, B_start + ngroups * d_state),
            "C_pre_conv": (C_start, C_start + ngroups * d_state),
            "dt_raw":     (dt_start, d_in_proj),
        }
        return bounds[part]

    @staticmethod
    def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
        """Inverse of F.softplus for y > 0: x such that softplus(x) == y.

        Mirrors the formula mamba_ssm itself uses to initialize dt_bias from a
        target dt (`inv_dt = dt + torch.log(-torch.expm1(-dt))`).
        """
        y = y.float()
        return y + torch.log(-torch.expm1(-y))

    def _register_hooks_slice(
        self, n_layer: int, hook_context: dict, part: str,
        capture_transform=None, patch_transform=None,
    ) -> List:
        """Register one forward hook per layer on mixer.in_proj, acting on a
        fixed column slice of its output (as given by `_in_proj_slice_bounds`).

        Used for the pre-conv components (B_pre_conv, C_pre_conv) directly, and
        for 'delta' by slicing the dt_raw columns and applying transforms:
          capture_transform(raw_slice, mixer) -> value to store as donor
          patch_transform(donor_value, mixer) -> value to write into the slice

        The hook_context contract matches `_register_hooks_dt_raw`/`_register_hooks_u`
        (mode/active_layers/donor/reg_positions/loc_positions/regular_len).
        """
        hooks = []

        for l in range(n_layer):
            def make_hook(layer_idx: int):
                def hook_fn(module, input, output):
                    mode = hook_context.get("mode", "off")
                    mixer = self.model.backbone.layers[layer_idx].mixer
                    start, end = self._in_proj_slice_bounds(mixer, part)

                    if mode == "capture":
                        if output.dim() == 3 and output.shape[1] > 1:
                            value = output[0, :, start:end].detach().clone()  # (T, width)
                            if capture_transform is not None:
                                value = capture_transform(value, mixer)
                            hook_context["donor"][layer_idx] = value

                    elif mode == "patch":
                        active = hook_context.get("active_layers", set())
                        if layer_idx not in active:
                            return

                        if output.dim() != 3 or output.shape[1] != hook_context.get("regular_len", -1):
                            return

                        donor = hook_context.get("donor", {})
                        if layer_idx not in donor:
                            return

                        reg_pos = hook_context.get("reg_positions", [])
                        loc_pos = hook_context.get("loc_positions", [])
                        if not reg_pos:
                            return

                        reg_idx = torch.tensor(reg_pos, device=output.device, dtype=torch.long)
                        loc_idx = torch.tensor(loc_pos, device=output.device, dtype=torch.long)
                        donor_value = donor[layer_idx][loc_idx]
                        if patch_transform is not None:
                            donor_value = patch_transform(donor_value, mixer)

                        patched = output.clone()
                        patched[0, reg_idx, start:end] = donor_value.to(patched.dtype)
                        return patched

                    elif mode == "scale":
                        active = hook_context.get("active_layers", set())
                        if layer_idx not in active:
                            return

                        if output.dim() != 3 or output.shape[1] != hook_context.get("regular_len", -1):
                            return

                        reg_pos = hook_context.get("reg_positions", [])
                        if not reg_pos:
                            return

                        reg_idx = torch.tensor(reg_pos, device=output.device, dtype=torch.long)
                        factors = self._scale_factor_tensor(hook_context, reg_pos, output.device)
                        # Read current slice values.
                        current = output[0, reg_idx, start:end]  # (len(reg_pos), width)
                        if capture_transform is not None:
                            # Convert to the semantic space (e.g. dt_raw → Δ), scale, then invert.
                            current_transformed = capture_transform(current.float(), mixer)
                            scaled_transformed = current_transformed * factors.unsqueeze(1)
                            scaled = patch_transform(scaled_transformed, mixer)
                        else:
                            scaled = current.float() * factors.unsqueeze(1)

                        patched = output.clone()
                        patched[0, reg_idx, start:end] = scaled.to(patched.dtype)
                        return patched

                    # mode == 'off': do nothing.

                return hook_fn

            mixer = self.model.backbone.layers[l].mixer
            h = mixer.in_proj.register_forward_hook(make_hook(l))
            hooks.append(h)

        return hooks

    def _register_hooks_delta(self, n_layer: int, hook_context: dict) -> List:
        """Patch Δ = softplus(dt_raw + dt_bias) by capturing the donor's Δ and
        inverting it back into dt_raw's logit space before writing it into the
        in_proj output. Since donor and recipient are the same model (only the
        prompt differs), dt_bias is identical on both sides, so this reconstructs
        the donor's Δ exactly once the recipient applies its own +dt_bias/softplus.
        """
        def capture_transform(dt_raw_slice, mixer):
            return F.softplus(dt_raw_slice.float() + mixer.dt_bias.float())

        def patch_transform(donor_delta, mixer):
            return self._inv_softplus(donor_delta) - mixer.dt_bias.float()

        return self._register_hooks_slice(
            n_layer, hook_context, "dt_raw",
            capture_transform=capture_transform, patch_transform=patch_transform,
        )

    def _install_post_conv_monkeypatch(self, hook_contexts: Dict[str, dict], scan_state: dict):
        """Monkeypatch mamba_chunk_scan_combined to capture/patch B and/or C
        *after* the causal conv1d + SiLU activation — i.e. the exact tensors
        consumed by the SSM scan.

        This is necessary because, unlike in_proj, the conv1d is invoked via a
        fused Triton kernel (causal_conv1d_fn) called directly inside
        mixer.forward rather than through the conv1d nn.Module, so there is no
        module boundary to attach a forward hook to. mamba_chunk_scan_combined
        is itself a plain function (not a module method), so we replace the
        name in mamba_ssm.modules.mamba2's module namespace for the duration of
        the run and restore it afterwards.

        Relies on mamba_chunk_scan_combined being called exactly once per layer,
        in layer order, within a single prefill forward pass (true for
        model.generate(..., cg=False): decode steps use `mixer.step()` /
        selective_state_update instead). `scan_state['counter']` tracks this
        call order and is reset to 0 before every generate() call by
        `_generate_with_hooks`.

        Only installed when 'B_post_conv' and/or 'C_post_conv' are requested;
        returns the original function so it can be restored in run()'s
        `finally` block.
        """
        original_fn = mamba2_module.mamba_chunk_scan_combined
        parts = [p for p in ("B_post_conv", "C_post_conv") if p in hook_contexts]

        def wrapped(x, dt, A, B, C, **kwargs):
            layer_idx = scan_state["counter"]
            scan_state["counter"] += 1
            tensors = {"B_post_conv": B, "C_post_conv": C}

            for part in parts:
                hook_context = hook_contexts[part]
                mode = hook_context.get("mode", "off")
                tensor = tensors[part]  # (batch, seqlen, ngroups, d_state)

                if mode == "capture":
                    if tensor.dim() == 4 and tensor.shape[1] > 1:
                        hook_context["donor"][layer_idx] = (
                            tensor[0].detach().clone().reshape(tensor.shape[1], -1)
                        )  # (T, ngroups * d_state)

                elif mode == "patch":
                    active = hook_context.get("active_layers", set())
                    if layer_idx not in active:
                        continue
                    if tensor.dim() != 4 or tensor.shape[1] != hook_context.get("regular_len", -1):
                        continue
                    donor = hook_context.get("donor", {})
                    if layer_idx not in donor:
                        continue
                    reg_pos = hook_context.get("reg_positions", [])
                    loc_pos = hook_context.get("loc_positions", [])
                    if not reg_pos:
                        continue

                    ngroups, d_state = tensor.shape[2], tensor.shape[3]
                    reg_idx = torch.tensor(reg_pos, device=tensor.device, dtype=torch.long)
                    loc_idx = torch.tensor(loc_pos, device=tensor.device, dtype=torch.long)
                    donor_value = donor[layer_idx][loc_idx].view(-1, ngroups, d_state).to(tensor.dtype)

                    patched = tensor.clone()
                    patched[0, reg_idx] = donor_value
                    tensors[part] = patched

                elif mode == "scale":
                    active = hook_context.get("active_layers", set())
                    if layer_idx not in active:
                        continue
                    if tensor.dim() != 4 or tensor.shape[1] != hook_context.get("regular_len", -1):
                        continue
                    reg_pos = hook_context.get("reg_positions", [])
                    if not reg_pos:
                        continue

                    ngroups, d_state = tensor.shape[2], tensor.shape[3]
                    reg_idx = torch.tensor(reg_pos, device=tensor.device, dtype=torch.long)
                    factors = self._scale_factor_tensor(hook_context, reg_pos, tensor.device)
                    patched = tensor.clone()
                    # factors: (len(reg_pos),) → broadcast over (ngroups, d_state).
                    patched[0, reg_idx] = (
                        tensor[0, reg_idx].float() * factors.view(-1, 1, 1)
                    ).to(tensor.dtype)
                    tensors[part] = patched

            return original_fn(x, dt, A, tensors["B_post_conv"], tensors["C_post_conv"], **kwargs)

        mamba2_module.mamba_chunk_scan_combined = wrapped
        return original_fn

    def _register_hooks(self, component: str, n_layer: int, n_heads: int, hook_context: dict) -> List:
        """Dispatch to the per-component hook registrar.

        'B_post_conv'/'C_post_conv' are not registered here as per-layer module
        hooks — they are installed once, jointly, via
        `_install_post_conv_monkeypatch` in run().
        """
        if component == "dt_raw":
            return self._register_hooks_dt_raw(n_layer, n_heads, hook_context)
        elif component == "u":
            return self._register_hooks_u(n_layer, hook_context)
        elif component == "delta":
            return self._register_hooks_delta(n_layer, hook_context)
        elif component in ("B_pre_conv", "C_pre_conv"):
            return self._register_hooks_slice(n_layer, hook_context, component)
        elif component in ("B_post_conv", "C_post_conv"):
            return []
        raise ValueError(f"Unknown patching-and-scaling component: {component!r}")

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _generate_with_hooks(
        self,
        prompt: "str | List[int]",
        max_new_tokens: int,
        hook_contexts: Dict[str, dict],
        modes: Dict[str, str],
        active_layers: Dict[str, Set[int]] = None,
    ) -> str:
        """Encode (or use pre-built ids), generate (cg=False to avoid
        CUDA-graph / hook conflict), decode.

        `prompt` is either prompt text (tokenized normally via _encode) or a
        pre-built list of token ids — used for the local side on SentencePiece
        name_start, whose token sequence is assembled from canonical segments
        rather than by tokenizing the rendered string (see _build_alignment);
        in that case the ids must be fed to the model as-is, since
        re-tokenizing the corresponding text would not reproduce them.
        GPT-NeoX name_start feeds the natural string, like name_end / global.

        `modes` gives the desired mode ('capture' | 'patch' | 'off') for each
        component's hook_context; components not in `modes` default to 'off'.
        `active_layers`, if given, sets the active layer set for the components
        being patched.
        """
        # Reset the shared scan-call counter (only meaningful when a
        # B_post_conv/C_post_conv monkeypatch is installed) so layer indices
        # recovered from call order start at 0 for this generate() call.
        if hasattr(self, "_scan_state"):
            self._scan_state["counter"] = 0

        for component, hook_context in hook_contexts.items():
            hook_context["mode"] = modes.get(component, "off")
            if active_layers is not None and component in active_layers:
                hook_context["active_layers"] = active_layers[component]

        if isinstance(prompt, str):
            input_ids, _ = self._encode([prompt])
        else:
            input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_new_tokens,
                cg=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = input_ids.shape[1]
        generated = out[0, prompt_len:].tolist()
        return self._decode(generated)

    def _extract_label(self, text: str) -> str | None:
        """Extract the answer from generated text using the dataset's label regex."""
        m = re.search(self._label_pattern, text)
        return m.group(0) if m else None

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self):
        if self.args.model not in MAMBA_MODELS:
            raise ValueError(f"PatchingAndScaling is only supported for mamba models: {MAMBA_MODELS}")

        print("#" * 80)
        print(
            f"{'#' * 12} GATE PATCHING [{self._component_tag}] "
            f"{self._patch_direction_label} "
            f"[{self._patching_and_scaling_entries}/{self._patching_and_scaling_entry_part}] "
            f"— {NUM_SAMPLES} SAMPLES {'#' * 12}"
        )
        print("#" * 80)

        self.model.eval()
        n_layer = self.model.config.n_layer
        n_heads = self.model.backbone.layers[0].mixer.nheads

        # Build sweep configs (only for the requested sweep types).
        sweeps = {sw: self._SWEEP_BUILDERS[sw](n_layer) for sw in self._sweep_types}

        # Accumulators: sweep_type -> label -> {"n_correct": int, "n_samples": int}.
        # All requested components are patched jointly, so there is a single
        # condition per (sweep_type, layer_range) rather than one per component.
        sweep_counts: Dict[str, Dict[str, dict]] = {
            sw: {label: {"n_correct": 0, "n_samples": 0} for label, _ in cfgs}
            for sw, cfgs in sweeps.items()
        }
        donor_correct = 0
        recipient_correct = 0
        total_samples = 0

        # Dataset — {phonebook,profbook}_local (either flag location), batch_size=1.
        dataset = self._get_dataset(self.args.patching_and_scaling_entries_per_prompt)
        max_new_tokens = dataset.max_new_tokens()
        self._label_pattern = dataset.label_regex_pattern()

        # Register persistent hooks — one hook_context per requested component.
        hook_contexts: Dict[str, dict] = {}
        hooks = []
        for component in self._components:
            hook_context = {
                "mode": "off",
                "donor": {},
                "active_layers": set(),
                "reg_positions": [],
                "loc_positions": [],
                "regular_len": 0,
            }
            hook_contexts[component] = hook_context
            hooks.extend(self._register_hooks(component, n_layer, n_heads, hook_context))

        # B_post_conv/C_post_conv have no module boundary to hook (conv1d is
        # invoked via a fused Triton kernel), so they're captured/patched via a
        # single joint monkeypatch of mamba_chunk_scan_combined instead.
        self._scan_state = {"counter": 0}
        original_scan_fn = None
        if any(c in ("B_post_conv", "C_post_conv") for c in self._components):
            original_scan_fn = self._install_post_conv_monkeypatch(hook_contexts, self._scan_state)

        off_modes = {component: "off" for component in self._components}
        capture_modes = {component: "capture" for component in self._components}
        action_modes = (
            {component: "scale" for component in self._components}
            if self._is_scaling
            else {component: "patch" for component in self._components}
        )

        try:
            for sample_idx in tqdm(range(NUM_SAMPLES), desc="PatchingAndScaling"):
                batch = dataset[0]
                local_prompt = batch["input"][0]
                prompt_parts = batch["prompt_parts"][0]
                label = batch["label"][0]
                target_entry_index = batch["target_index"][0]

                # Build alignment (also tracks target-entry positions for scaling mode).
                try:
                    alignment = self._build_alignment(
                        prompt_parts, local_prompt, target_entry_index=target_entry_index
                    )
                except AssertionError as e:
                    print(f"[sample {sample_idx}] alignment error, skipping: {e}")
                    continue

                if self._is_scaling:
                    recipient_prompt = self._prompt_for_side(alignment, self._patch_recipient)
                    self._configure_hook_contexts_scaling(hook_contexts, alignment)
                else:
                    donor_prompt, recipient_prompt = self._prompts_from_alignment(alignment)
                    self._configure_hook_contexts(hook_contexts, alignment)

                    # Donor capture (all components at once).
                    self._generate_with_hooks(
                        donor_prompt, max_new_tokens, hook_contexts, modes=capture_modes
                    )

                    # Unpatched reference accuracy on donor prompt.
                    donor_pred_text = self._generate_with_hooks(
                        donor_prompt, max_new_tokens, hook_contexts, modes=off_modes
                    )
                    donor_pred = self._extract_label(donor_pred_text)
                    donor_correct += int(donor_pred == label)

                # Unpatched reference accuracy on recipient prompt.
                recipient_pred_text = self._generate_with_hooks(
                    recipient_prompt, max_new_tokens, hook_contexts, modes=off_modes
                )
                recipient_pred = self._extract_label(recipient_pred_text)
                recipient_correct += int(recipient_pred == label)

                total_samples += 1

                # Patching / scaling sweeps — every requested component is acted on
                # jointly (same layer_set) in a single generate call per layer-range.
                for sw, configs in sweeps.items():
                    for label_str, layer_set in configs:
                        active_layers = {component: layer_set for component in self._components}
                        acted_pred_text = self._generate_with_hooks(
                            recipient_prompt, max_new_tokens, hook_contexts,
                            modes=action_modes, active_layers=active_layers,
                        )
                        acted_pred = self._extract_label(acted_pred_text)
                        counts = sweep_counts[sw][label_str]
                        counts["n_samples"] += 1
                        counts["n_correct"] += int(acted_pred == label)

                # Free donor cache each sample.
                for hook_context in hook_contexts.values():
                    hook_context["donor"] = {}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            for h in hooks:
                h.remove()
            if original_scan_fn is not None:
                mamba2_module.mamba_chunk_scan_combined = original_scan_fn

        # Compute reference accuracies.
        donor_accuracy = donor_correct / max(total_samples, 1) if not self._is_scaling else None
        recipient_accuracy = recipient_correct / max(total_samples, 1)

        # Build results structure.
        sweep_results = {}
        for sw, configs in sweeps.items():
            sweep_results[sw] = []
            for label_str, _ in configs:
                counts = sweep_counts[sw][label_str]
                n = counts["n_samples"]
                nc = counts["n_correct"]
                sweep_results[sw].append({
                    "layer_range": label_str,
                    "n_samples": n,
                    "n_correct": nc,
                    "accuracy": nc / max(n, 1),
                })

        self.results = {
            "n_samples": total_samples,
            "entries_per_prompt": self.args.patching_and_scaling_entries_per_prompt,
            "local_dataset_flag_location": self.args.local_dataset_flag_location,
            "local_dataset_flag_entries": self.args.local_dataset_flag_entries,
            "dataset_family": self._dataset_family,
            "patched_components": self._components,
            "patching_and_scaling_entries": self._patching_and_scaling_entries,
            "patching_and_scaling_entry_part": self._patching_and_scaling_entry_part,
            "patch_donor": self._patch_donor,
            "patch_recipient": self._patch_recipient,
            "patch_scale": self._patch_scale if self._is_scaling else None,
            "donor_dataset": self._donor_dataset,
            "recipient_dataset": self._recipient_dataset,
            "patch_direction": self._patch_direction_label,
            "donor_accuracy": donor_accuracy,
            "recipient_accuracy": recipient_accuracy,
            "sweeps": sweep_results,
        }

        if self._is_scaling:
            print(
                f"\nReference accuracy: recipient ({self._recipient_dataset})={recipient_accuracy:.4f}  "
                f"(scale x{self._patch_scale}: target entries ×{self._patch_scale}, "
                f"other entries ÷{self._patch_scale})"
            )
        else:
            print(
                f"\nReference accuracies: donor ({self._donor_dataset})={donor_accuracy:.4f}  "
                f"recipient ({self._recipient_dataset})={recipient_accuracy:.4f}"
            )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def save_plots(self):
        for sw in self._sweep_types:
            self._plot_sweep(sw)

    def _plot_sweep(self, sweep_type: str):
        rows = self.results["sweeps"][sweep_type]
        labels = [r["layer_range"] for r in rows]
        accuracies = [r["accuracy"] for r in rows]
        recipient_acc = self.results["recipient_accuracy"]
        recipient_ds = self.results["recipient_dataset"]
        patch_direction = self.results["patch_direction"]

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.4), 5))
        x = range(len(labels))
        bar_label = (
            f"Recipient scaled ({recipient_ds})" if self._is_scaling
            else f"Recipient patched ({recipient_ds})"
        )
        ax.bar(x, accuracies, color="steelblue", alpha=0.8, label=bar_label)
        if not self._is_scaling:
            donor_acc = self.results["donor_accuracy"]
            donor_ds = self.results["donor_dataset"]
            ax.axhline(donor_acc, color="#d62728", linestyle="--", linewidth=1.5,
                       label=f"Donor unpatched ({donor_ds}) = {donor_acc:.3f}")
        ax.axhline(recipient_acc, color="#2ca02c", linestyle="--", linewidth=1.5,
                   label=f"Recipient unpatched ({recipient_ds}) = {recipient_acc:.3f}")

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=90, fontsize=max(6, 8 - len(labels) // 20))
        ax.set_ylim(0, 1)
        xlabel = "Patched layer" if sweep_type == "isolated" else "Patched layer range"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy on recipient prompt")
        sweep_title = SWEEP_TITLES.get(sweep_type, sweep_type)
        component_title = " + ".join(COMPONENT_TITLES.get(c, c) for c in self._components)
        model_tag = f"{self.args.model}, {self.args.task}"
        action_word = "Scaling" if self._is_scaling else "Patching"
        ax.set_title(
            f"{component_title} {action_word} — {patch_direction} "
            f"[{self._patching_and_scaling_entries}/{self._patching_and_scaling_entry_part}]\n"
            f"{sweep_title} ({model_tag}, n={self.results['n_samples']})"
        )
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = self._get_plots_path() + f"{self._variant_prefix}sweep_{sweep_type}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot to {out_path}")
