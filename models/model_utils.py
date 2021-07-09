from models import GGNN
from dataProcessing import matrixcompeltionGraphDataset, matrixcompletionMake_task_input

def name_to_dataset_class(name: str, args):
    name = name.lower()
    #return classname, appendix attribute, make_task_input_function,
    if name in ["matrixrandom"]:
        return matrixcompeltionGraphDataset, {}, matrixcompletionMake_task_input
    raise ValueError("Unkown dataset name '%s'" % name)


def name_to_model_class(name: str, args):
    name = name.lower()
    if name in ["ggnn"]:
        return GGNN, {}
    raise ValueError("Unkown model name '%s'" % name)