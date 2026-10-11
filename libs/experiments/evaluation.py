"""Evaluation experiment.

Sweeps phonebook / profbook length and reports generation accuracy of the
loaded model on the chosen task.

Requirements
------------
Mamba models (MambaLMHeadModel.generate ignores attention_mask and treats pad
tokens as real input) require --eval-batch-size 1. Transformers may use a
larger batch.

Configuration
-------------
  --min-length / --max-length / --length-step  (defaults 20 / 200 / 20)
      Phonebook lengths to evaluate.
  --eval-batch-size / --eval-num-batches       (defaults 1 / 96)
      Batching of the generated dataset.
  --dataset, --task, --model
      Which prompts and model to score.
  --local-dataset-flag-entries
      When --dataset ends with '_local', this value is prepended to output
      filenames via _variant_prefix.

Protocol per length
-------------------
1. Build a dataset of that many entries.
2. For each eval batch: encode, generate max_new_tokens, decode the
   continuation, and search it with dataset.label_regex_pattern().
3. If the regex matches, compare the match to the ground-truth label.
   If it does not match, count the example toward no_phone_ratio and
   not toward accuracy.

Metric
------
accuracy = (# exact label matches) / (# examples).
no_phone_ratio = (# regex misses) / (# examples).
Regex-miss examples stay in the denominator of both ratios.

Outputs
-------
results.json keys: accuracy_list, no_phone_ratio_list, lengths.
Plot: {variant_prefix}evaluation.png (accuracy vs length).
"""

import re

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from libs.utils import print_space
from libs.experiments.experiment import Experiment
from libs.models.models import MAMBA_MODELS


class Evaluation(Experiment):
	def __init__(self, args, model, tokenizer):
		super().__init__(args, model, tokenizer)
		if self.args.model in MAMBA_MODELS and self.args.eval_batch_size != 1:
			raise ValueError(
				f"Mamba evaluation requires --eval-batch-size 1 (got {self.args.eval_batch_size}). "
				"MambaLMHeadModel.generate() ignores attention_mask and treats pad tokens as "
				"real input, so batched eval is biased."
			)

	def run(self):
		accuracy_list = []
		no_phone_ratio_list = []
		completed_lengths = []
		lengths = list(range(self.args.min_length, self.args.max_length + 1, self.args.length_step))

		try:
			for length in lengths:
				print(f"{'#' * 80}")
				print(f"{'#' * 22} PHONE BOOK LENGTH: {length} {'#' * 22}")
				print(f"{'#' * 80}")

				accuracy, no_phone_ratio = self.run_length(length)
				print(f"length {length}: accuracy - {accuracy:.3f} ; no_phone_ratio - {no_phone_ratio:.3f}")
				accuracy_list.append(accuracy)
				no_phone_ratio_list.append(no_phone_ratio)
				completed_lengths.append(length)
				self.results = {
					"accuracy_list": accuracy_list,
					"no_phone_ratio_list": no_phone_ratio_list,
					"lengths": completed_lengths,
				}
				self.save_results()
				print_space()
		except torch.OutOfMemoryError:
			if self.results:
				self.save_results()
				print(
					f"CUDA OOM at length {lengths[len(completed_lengths)]}; "
					f"saved partial results for lengths {completed_lengths}"
				)
			raise

	@property
	def _variant_prefix(self):
		if self.args.dataset.endswith("_local"):
			return f"{self.args.local_dataset_flag_entries}_"
		return ""

	def save_plots(self):
		accuracy_list = self.results["accuracy_list"]
		lengths = self.results["lengths"]
		plt.figure(figsize=(8, 5))
		plt.plot(lengths, accuracy_list, marker='o', label='Accuracy')
		plt.xlabel('Length')
		plt.ylabel('Accuracy')
		plt.title(f"{self.args.model}'s {self.args.task} Accuracy vs. Phone Book Length")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		plt.savefig(self._get_plots_path() + f"{self._variant_prefix}evaluation.png")
		plt.close()

	def run_length(self, length):
		self.model.eval()
		dataset = self._get_dataset(length)

		sum_accuracy = 0.0
		no_match_count = 0.0
		num_examples = 0

		for batch_idx in tqdm(range(self.args.eval_num_batches), desc="Evaluating"):
			batch = dataset[batch_idx]

			inputs = batch["input"]
			labels = batch["label"]
			input_ids, attention_mask = self._encode(inputs)

			with torch.inference_mode():
				sequences = self._generate(input_ids, attention_mask, max_new_tokens=dataset.max_new_tokens())
			sequences = sequences.detach().to("cpu")
			prompt_len = int(input_ids.shape[1])

			for idx, gt in enumerate(labels):
				generated_tokens = sequences[idx, prompt_len:]
				prediction = self._decode(generated_tokens.tolist())
				match = re.search(dataset.label_regex_pattern(), prediction)
				if match:
					prediction = match.group(0)
					sum_accuracy += int(gt == prediction)
				else:
					no_match_count += 1

				num_examples += 1

			del input_ids, attention_mask
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

		mean_accuracy = sum_accuracy / max(num_examples, 1)
		no_phone_ratio = no_match_count / max(num_examples, 1)
		return mean_accuracy, no_phone_ratio
