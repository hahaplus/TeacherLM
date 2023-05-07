from pprint import pprint
from typing import List, Union

import lm_eval.base

from . import mmlu
from . import creak
from . import ecqa
from . import strategy_qa

########################################
# All tasks
########################################


TASK_REGISTRY = {
    "abstract_algebra": mmlu.abstract_algebra,
    "astronomy": mmlu.astronomy,
    "college_biology": mmlu.college_biology,
    "college_chemistry": mmlu.college_chemistry,
    "college_computer_science": mmlu.college_computer_science,
    "college_mathematics": mmlu.college_mathematics,
    "college_physics": mmlu.college_physics,
    "computer_security": mmlu.computer_security,
    "conceptual_physics": mmlu.conceptual_physics,
    "electrical_engineering": mmlu.electrical_engineering,
    "elementary_mathematics": mmlu.elementary_mathematics,
    "high_school_biology": mmlu.high_school_biology,
    "high_school_chemistry": mmlu.high_school_chemistry,
    "high_school_computer_science": mmlu.high_school_computer_science,
    "high_school_mathematics": mmlu.high_school_mathematics,
    "high_school_physics": mmlu.high_school_physics,
    "high_school_statistics": mmlu.high_school_statistics,
    "machine_learning": mmlu.machine_learning,
    "formal_logic": mmlu.formal_logic,
    "high_school_european_history": mmlu.high_school_european_history,
    "high_school_us_history": mmlu.high_school_us_history,
    "high_school_world_history": mmlu.high_school_world_history,
    "international_law": mmlu.international_law,
    "jurisprudence": mmlu.jurisprudence,
    "logical_fallacies": mmlu.logical_fallacies,
    "moral_disputes": mmlu.moral_disputes,
    "moral_scenarios": mmlu.moral_scenarios,
    "philosophy": mmlu.philosophy,
    "prehistory": mmlu.prehistory,
    "professional_law": mmlu.professional_law,
    "world_religions": mmlu.world_religions,
    "econometrics": mmlu.econometrics,
    "high_school_geography": mmlu.high_school_geography,
    "high_school_government_and_politics": mmlu.high_school_government_and_politics,
    "high_school_macroeconomics": mmlu.high_school_macroeconomics,
    "high_school_microeconomics": mmlu.high_school_microeconomics,
    "high_school_psychology": mmlu.high_school_psychology,
    "human_sexuality": mmlu.human_sexuality,
    "professional_psychology": mmlu.professional_psychology,
    "public_relations": mmlu.public_relations,
    "security_studies": mmlu.security_studies,
    "sociology": mmlu.sociology,
    "us_foreign_policy": mmlu.us_foreign_policy,
    "anatomy": mmlu.anatomy,
    "business_ethics": mmlu.business_ethics,
    "clinical_knowledge": mmlu.clinical_knowledge,
    "college_medicine": mmlu.college_medicine,
    "global_facts": mmlu.global_facts,
    "human_aging": mmlu.human_aging,
    "management": mmlu.management,
    "marketing": mmlu.marketing,
    "medical_genetics": mmlu.medical_genetics,
    "miscellaneous": mmlu.miscellaneous,
    "nutrition": mmlu.nutrition,
    "professional_accounting": mmlu.professional_accounting,
    "professional_medicine": mmlu.professional_medicine,
    "virology": mmlu.virology,
    "creak": creak.CREAK,
    "ecqa": ecqa.ECQA,
    "strategy_qa": strategy_qa.StrategyQA
}


ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
