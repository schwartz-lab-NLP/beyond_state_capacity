import argparse

from libs.experiments.experiment_factory import EXPERIMENTS_MAP
from libs.models.models import ALL_MODELS


def parse_args():
	parser = argparse.ArgumentParser()
	# General
	parser.add_argument('--debug', action='store_true', help="Debug mode")
	parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument('--experiment', choices=EXPERIMENTS_MAP.keys(), type=str, default="evaluation",
						help="Which experiment to run.")
	parser.add_argument('--skip-collection', action='store_true',
						help="Skip the expensive first phase and resume from a prior run's "
							 "artifacts (requires --run-sha).")
	parser.add_argument('--run-sha', type=str, default=None,
						help="Run SHA subdirectory under raw_results/plots. Required with "
							 "--skip-collection for experiments that use run SHA subdirs")

	# Model
	parser.add_argument('--model', choices=ALL_MODELS, type=str, default="mamba_nvidia",
						help="Which model to load and evaluate.")

	# Dataset
	parser.add_argument('--task', choices=["prefix", "suffix", "both", "misleading_prefix", "no_query"], type=str, default="prefix",
						help="Where in the prompt the query for the target entry appears.")
	parser.add_argument('--dataset', choices=["phonebook", "phonebook_local", "profbook", "profbook_local"], type=str, default="phonebook",
						help="Which synthetic dataset to generate prompts from.")

	# Local Datasets
	parser.add_argument('--local-dataset-flag-location', choices=["name_start", "name_end"], default="name_start",
						help="Relevant only to the phonebook_local/profbook_local datasets: "
							 "'name_start' places the flag at the start of the entry name, "
							 "'name_end' places the flag at the end of the entry name.")
	parser.add_argument('--local-dataset-flag-entries', choices=["target", "skip", "target_and_skip"], default="target",
						help="Relevant only to the phonebook_local/profbook_local datasets: "
							 "'target' flags only the target entry, "
							 "'skip' flags all entries except the target entry, "
							 "'target_and_skip' flags all entries.")

	# Evaluation Experiment
	parser.add_argument('--min-length', default=20, type=int, help="Minimum entries in a sample")
	parser.add_argument('--max-length', default=200, type=int, help="Maximum entries in a sample")
	parser.add_argument('--length-step', default=20, type=int, help="Step size for the sample length")
	parser.add_argument('--eval-batch-size', default=1, type=int, help="Size of the batch size for evaluation.")
	parser.add_argument('--eval-num-batches', default=500, type=int, help="Number of batches for the evaluation to compute mean")

	# SelectionDistinction Experiment
	parser.add_argument('--selectivity-component',
						choices=['delta', 'dt_raw', 'B_norm', 'B_norm_post_conv', 'forget_factor'],
						default='dt_raw',
						help="What to track in the selection_distinction and gate_probing "
							 "experiments: "
							 "'delta' tracks Δ_t = softplus(dt_logit + dt_bias), the full "
							 "discretisation step-size used in the SSM recurrence (default); "
							 "'dt_raw' tracks only the input-dependent logit (before bias and softplus), "
							 "isolating the token's direct contribution to selectivity; "
							 "'B_norm' tracks the L2 norm of the input-dependent B projection (pre-conv) "
							 "per group, measuring token-local write-strength into the SSM state; "
							 "'B_norm_post_conv' is the same norm but computed after simulating the "
							 "causal conv1d step, reflecting the effective write-strength that actually "
							 "reaches the SSM (includes temporal smoothing from the last d_conv tokens); "
							 "'forget_factor' tracks dA = exp(Δ · A), the state retention coefficient "
							 "combining the input-dependent Δ with the learned per-head decay rate A. "
							 "gate_probing supports only 'dt_raw' and 'delta'.")
	parser.add_argument('--selection-distinction-entries-per-prompt',
						type=int, default=150,
						help="Relevant only to the selection_distinction experiment: "
							 "number of directory entries in each prompt.")

	# RepresentationProbing Experiment
	parser.add_argument('--representation-probing-entries-per-prompt',
						type=int, default=150,
						help="Relevant only to the representation_probing experiment: "
							 "number of directory entries in each prompt.")
	parser.add_argument('--representation-probing-entry-part',
						choices=['name_first_token', 'number_first_token', 'name_last_token', 'number_last_token'],
						default='number_last_token',
						help="Relevant to the representation_probing and gate_probing experiments: "
							 "'name_first_token' uses the first token of the entry name, "
							 "'number_first_token' uses the first token of the entry number, "
							 "'name_last_token' uses the last token of the entry name, "
							 "'number_last_token' uses the last token of the entry number.")

	# GateProbing Experiment
	parser.add_argument('--gate-probing-entries-per-prompt',
						type=int, default=150,
						help="Relevant only to the gate_probing experiment: "
							 "number of directory entries in each prompt.")

	# StateProbing Experiment
	parser.add_argument('--state-projection', choices=["none", "jl"], type=str, default="none",
						help="Relevant only to the state_probing experiment: how to reduce the "
							 "dimensionality of the collected SSM states before probing. "
							 "'none' (default) stores/probes the raw, unprojected states; "
							 "'jl' applies a fixed Gaussian Johnson-Lindenstrauss random "
							 "projection down to --state-projection-dim dimensions.")
	parser.add_argument('--state-projection-dim', type=int, default=4096,
						help="Relevant only to the state_probing experiment when "
							 "--state-projection=jl: target dimension (k) of the JL projection.")
	parser.add_argument('--state-probing-num-gpus', choices=[1, 2], type=int, default=1,
						help="Relevant only to the state_probing experiment: number of GPUs to use. "
							 "1 (default) runs the Mamba backbone and the online probe bank "
							 "(replay buffer, ProbeBank, optimizer) on the same GPU. "
							 "2 places the Mamba backbone on cuda:0 and the probe bank on cuda:1, "
							 "freeing enough memory to train full-size (unprojected) raw-mode probes "
							 "with a larger replay buffer/batch size. Requires 2 visible CUDA devices.")
	parser.add_argument('--state-probing-reservoir-slots', type=int, default=None,
						help="Relevant only to the state_probing experiment: number of "
							 "reservoir slots per profession class. Raw mode stores the reservoir "
							 "in CPU RAM (~2.35 GB per slot at 1M-dim x 56 layers fp16); default "
							 "is 24 slots (~56 GB) in raw mode, or 512 slots in jl mode. Increase "
							 "only if sbatch --mem allows it (e.g. 64 slots ~150 GB needs ~200 GB --mem).")
	parser.add_argument('--state-probing-entries-per-prompt',
						type=int, default=150,
						help="Relevant only to the state_probing experiment: "
							 "number of directory entries in each prompt.")

	# PatchingAndScaling Experiment
	parser.add_argument('--patching-and-scaling-entries-per-prompt',
						type=int, default=150,
						help="Relevant only to the patching_and_scaling, gate_patching_for_dt_diff, "
							 "and layer_skipping experiments: number of directory "
							 "entries in each prompt.")
	parser.add_argument('--patching-and-scaling-components',
						choices=['u', 'dt_raw', 'delta', 'B_pre_conv', 'B_post_conv',
								 'C_pre_conv', 'C_post_conv'],
						nargs='+', default=['delta'],
						help="Relevant only to the patching_and_scaling experiment: which mixer "
							 "component(s) to activation-patch. 'dt_raw' patches the "
							 "input-dependent discretization logit (pre-bias, pre-softplus); "
							 "'u' patches the full input representation fed into "
							 "mixer.in_proj; 'delta' patches Δ = softplus(dt_raw + dt_bias), "
							 "the actual discretization step-size used in the scan (implemented "
							 "by inverting the donor's Δ back into dt_raw's logit space, since "
							 "donor and recipient share the same dt_bias); 'B_pre_conv'/"
							 "'C_pre_conv' patch the input-dependent B/C projections before the "
							 "causal conv1d; 'B_post_conv'/'C_post_conv' patch B/C after the "
							 "conv1d + SiLU activation, i.e. the exact tensors consumed by the "
							 "SSM scan. Accepts multiple values (e.g. --patching-and-scaling-components "
							 "dt_raw u) to run the patching sweep for each component in the same "
							 "run.")
	parser.add_argument('--patching-and-scaling-entries',
						choices=['all_entries', 'target_entry', 'non_target_entries'],
						default='target_entry',
						help="Relevant only to the patching_and_scaling experiment: which entry "
							 "tokens to patch (or scale). 'target_entry' (default) patches "
							 "only the target entry's name/value tokens; 'non_target_entries' "
							 "patches only distractor entries; 'all_entries' patches every "
							 "entry's tokens.")
	parser.add_argument('--patching-and-scaling-entry-part',
						choices=['name', 'number', 'both', 'all'],
						default='number',
						help="Relevant only to the patching_and_scaling experiment: which part of the entry to patch. "
							 "'name' patches the name part of the entry, "
							 "'number' patches the number part of the entry, "
							 "'both' patches both the name and the number of the entry, "
							 "'all' patches the entire entry (name, colon, value with leading space, and newline).")
	parser.add_argument('--sweep', choices=['isolated', 'cumulative', 'reverse-cumulative', 'all'], nargs='+',
						default=['cumulative'],
						help="Relevant only to the patching_and_scaling experiment: which layer-range sweep(s) to run. "
							 "'isolated' patches each layer on its own; 'cumulative' patches [1-k] for "
							 "each k; 'reverse-cumulative' patches [k-n] for each k; 'all' is a single "
							 "run that patches every layer at once.")
	parser.add_argument('--patch-donor', choices=['local', 'global', 'scaling'],
						default='local',
						help="Relevant only to the patching_and_scaling experiment: which prompt "
							 "supplies the donor activations. 'local' captures from the "
							 "flag-prefixed local prompt (e.g. phonebook_local); 'global' "
							 "captures from the matching global prompt (e.g. phonebook). "
							 "Must differ from --patch-recipient. 'scaling' disables donor "
							 "capture entirely and instead scales the component in-place on "
							 "the recipient prompt by --patch-scale at target-entry token "
							 "positions and by 1/--patch-scale at all other patched entries' "
							 "positions.")
	parser.add_argument('--patch-scale', type=float, default=2.0,
						help="Relevant only to the patching_and_scaling experiment when "
							 "--patch-donor=scaling: factor to multiply the patched "
							 "component by at target-entry token positions (and divide "
							 "by at all other patched entries' positions).")
	parser.add_argument('--patch-recipient', choices=['local', 'global'],
						default='global',
						help="Relevant only to the patching_and_scaling experiment: which prompt "
							 "receives the patched activations. 'global' (default) patches "
							 "into the matching global prompt (e.g. phonebook); 'local' "
							 "patches into the flag-prefixed local prompt (e.g. "
							 "phonebook_local). Must differ from --patch-donor.")



def print_args(args):
	for arg in args.__dict__:
		print(f"{arg}: {args.__dict__[arg]}")

