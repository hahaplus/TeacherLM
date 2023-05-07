#!/bin/sh


python main.py \
    --model bloom \
    --model_args pretrained=${1} \
    --device 0 \
    --batch_size 1 \
    --no_cache \
    --tasks abstract_algebra,astronomy,college_biology,college_chemistry,college_computer_science,college_mathematics,college_physics,computer_security,conceptual_physics,electrical_engineering,elementary_mathematics,high_school_biology,high_school_chemistry,high_school_computer_science,high_school_mathematics,high_school_physics,high_school_statistics,machine_learning,formal_logic,high_school_european_history,high_school_us_history,high_school_world_history,international_law,jurisprudence,logical_fallacies,moral_disputes,moral_scenarios,philosophy,prehistory,professional_law,world_religions,econometrics,high_school_geography,high_school_government_and_politics,high_school_macroeconomics,high_school_microeconomics,high_school_psychology,human_sexuality,professional_psychology,public_relations,security_studies,sociology,us_foreign_policy,anatomy,business_ethics,clinical_knowledge,college_medicine,global_facts,human_aging,management,marketing,medical_genetics,miscellaneous,nutrition,professional_accounting,professional_medicine,virology