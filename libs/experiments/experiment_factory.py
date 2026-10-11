"""Maps --experiment names to Experiment subclasses.

get_experiment() constructs the class listed in EXPERIMENTS_MAP for args.experiment.
"""

from libs.experiments.experiment import Experiment
from libs.experiments.evaluation import Evaluation
from libs.experiments.selection_distinction import SelectionDistinction
from libs.experiments.representation_probing import RepresentationProbing
from libs.experiments.state_probing import StateProbing
from libs.experiments.patching_and_scaling import PatchingAndScaling
from libs.experiments.gate_probing import GateProbing


EXPERIMENTS_MAP = {
	"evaluation": Evaluation,
	"selection_distinction": SelectionDistinction,
	"representation_probing": RepresentationProbing, 
	"state_probing": StateProbing, 
	"patching_and_scaling": PatchingAndScaling,
	"gate_probing": GateProbing, 
}


def get_experiment(args, model, tokenizer) -> Experiment:
	return EXPERIMENTS_MAP[args.experiment](args, model, tokenizer)
