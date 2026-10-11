import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys

from libs.models.models import get_model, print_model, get_tokenizer
from libs.experiments.experiment_factory import get_experiment
from libs.experiments.experiment import Experiment
from libs.args import parse_args, print_args
from libs.utils import print_header, print_space, seed_everything, check_cuda_driver_consistency, warm_up_mamba_kernels


def main():
	if not check_cuda_driver_consistency():
		sys.exit(1)
	print_header("ARGS")
	args = parse_args()
	seed_everything(args.seed)
	if args.debug:
		os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
	print_args(args)
	print_space()

	print_header("MODEL")
	model = get_model(args)
	tokenizer = get_tokenizer(args)
	print_model(model)
	warm_up_mamba_kernels(model, tokenizer)
	print_space()

	experiment: Experiment = get_experiment(args, model, tokenizer)
	experiment.run()
	experiment.save_results()
	experiment.save_plots()


if __name__ == "__main__":
	main()
