import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
import json
import re
import datasets
 
dataset_dir = 'lm_eval/datasets/mmlu/'

class Mmlu(Task):
    VERSION = 0
    DATASET_PATH = dataset_dir
    DATASET_NAME = "mmlu"
        
    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["validation"]
        # return datasets.Dataset.from_dict(self.dataset["validation"][:20])
    
    def doc_to_text(self, doc):
        sig = ['.',',','?','!','"']

        if doc['question'][-1] not in sig:
            doc['question'] += '.'
            
        doc['question'] = doc['question'].replace('\n',' ')
        doc['A'] = doc['A'].replace('\n',' ')
        doc['B'] = doc['B'].replace('\n',' ')
        doc['C'] = doc['C'].replace('\n',' ')
        doc['D'] = doc['D'].replace('\n',' ')

        field = f"The following are multiple choice questions (with answers) about {doc['task'].replace('_',' ')}. "
        optinon_str = f"(A) {doc['A']} (B) {doc['B']} (C) {doc['C']} (D) {doc['D']}"
        
        ret_ctx = ''
        ret_ctx += field
        ret_ctx += f"Q: {doc['question']} {optinon_str} A: The answer is ("
        
        return ret_ctx
    
    def construct_requests(self, doc, ctx):
        
        ll_A, _ = rf.loglikelihood(ctx, "A")
        ll_B, _ = rf.loglikelihood(ctx, "B")
        ll_C, _ = rf.loglikelihood(ctx, "C")
        ll_D, _ = rf.loglikelihood(ctx, "D")
        requests = [ll_A, ll_B, ll_C, ll_D]
        return requests
    
    
    def doc_to_target(self, doc):
        return " "
        
    def process_results(self, doc, results):        

        gold = doc["answer"]
        option_term=['A','B','C','D','E','F','G','H']
        pred = np.argmax(results)
        acc = 1.0 if option_term[pred] == gold else 0.0
        return {"acc": acc}
    
    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
    
class abstract_algebra(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'abstract_algebra'
    DATASET_NAME = 'abstract_algebra'
class astronomy(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'astronomy'
    DATASET_NAME = 'astronomy'
class college_biology(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'college_biology'
    DATASET_NAME = 'college_biology'
class college_chemistry(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'college_chemistry'
    DATASET_NAME = 'college_chemistry'
class college_computer_science(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'college_computer_science'
    DATASET_NAME = 'college_computer_science'
class college_mathematics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'college_mathematics'
    DATASET_NAME = 'college_mathematics'
class college_physics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'college_physics'
    DATASET_NAME = 'college_physics'
class computer_security(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'computer_security'
    DATASET_NAME = 'computer_security'
class conceptual_physics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'conceptual_physics'
    DATASET_NAME = 'conceptual_physics'
class electrical_engineering(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'electrical_engineering'
    DATASET_NAME = 'electrical_engineering'
class elementary_mathematics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'elementary_mathematics'
    DATASET_NAME = 'elementary_mathematics'
class high_school_biology(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_biology'
    DATASET_NAME = 'high_school_biology'
class high_school_chemistry(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_chemistry'
    DATASET_NAME = 'high_school_chemistry'
class high_school_computer_science(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_computer_science'
    DATASET_NAME = 'high_school_computer_science'
class high_school_mathematics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_mathematics'
    DATASET_NAME = 'high_school_mathematics'
class high_school_physics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_physics'
    DATASET_NAME = 'high_school_physics'
class high_school_statistics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_statistics'
    DATASET_NAME = 'high_school_statistics'
class machine_learning(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'machine_learning'
    DATASET_NAME = 'machine_learning'
class formal_logic(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'formal_logic'
    DATASET_NAME = 'formal_logic'
class high_school_european_history(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_european_history'
    DATASET_NAME = 'high_school_european_history'
class high_school_us_history(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_us_history'
    DATASET_NAME = 'high_school_us_history'
class high_school_world_history(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_world_history'
    DATASET_NAME = 'high_school_world_history'
class international_law(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'international_law'
    DATASET_NAME = 'international_law'
class jurisprudence(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'jurisprudence'
    DATASET_NAME = 'jurisprudence'
class logical_fallacies(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'logical_fallacies'
    DATASET_NAME = 'logical_fallacies'
class moral_disputes(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'moral_disputes'
    DATASET_NAME = 'moral_disputes'
class moral_scenarios(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'moral_scenarios'
    DATASET_NAME = 'moral_scenarios'
class philosophy(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'philosophy'
    DATASET_NAME = 'philosophy'
class prehistory(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'prehistory'
    DATASET_NAME = 'prehistory'
class professional_law(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'professional_law'
    DATASET_NAME = 'professional_law'
class world_religions(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'world_religions'
    DATASET_NAME = 'world_religions'
class econometrics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'econometrics'
    DATASET_NAME = 'econometrics'
class high_school_geography(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_geography'
    DATASET_NAME = 'high_school_geography'
class high_school_government_and_politics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_government_and_politics'
    DATASET_NAME = 'high_school_government_and_politics'
class high_school_macroeconomics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_macroeconomics'
    DATASET_NAME = 'high_school_macroeconomics'
class high_school_microeconomics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_microeconomics'
    DATASET_NAME = 'high_school_microeconomics'
class high_school_psychology(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'high_school_psychology'
    DATASET_NAME = 'high_school_psychology'
class human_sexuality(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'human_sexuality'
    DATASET_NAME = 'human_sexuality'
class professional_psychology(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'professional_psychology'
    DATASET_NAME = 'professional_psychology'
class public_relations(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'public_relations'
    DATASET_NAME = 'public_relations'
class security_studies(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'security_studies'
    DATASET_NAME = 'security_studies'
class sociology(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'sociology'
    DATASET_NAME = 'sociology'
class us_foreign_policy(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'us_foreign_policy'
    DATASET_NAME = 'us_foreign_policy'
class anatomy(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'anatomy'
    DATASET_NAME = 'anatomy'
class business_ethics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'business_ethics'
    DATASET_NAME = 'business_ethics'
class clinical_knowledge(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'clinical_knowledge'
    DATASET_NAME = 'clinical_knowledge'
class college_medicine(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'college_medicine'
    DATASET_NAME = 'college_medicine'
class global_facts(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'global_facts'
    DATASET_NAME = 'global_facts'
class human_aging(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'human_aging'
    DATASET_NAME = 'human_aging'
class management(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'management'
    DATASET_NAME = 'management'
class marketing(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'marketing'
    DATASET_NAME = 'marketing'
class medical_genetics(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'medical_genetics'
    DATASET_NAME = 'medical_genetics'
class miscellaneous(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'miscellaneous'
    DATASET_NAME = 'miscellaneous'
class nutrition(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'nutrition'
    DATASET_NAME = 'nutrition'
class professional_accounting(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'professional_accounting'
    DATASET_NAME = 'professional_accounting'
class professional_medicine(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'professional_medicine'
    DATASET_NAME = 'professional_medicine'
class virology(Mmlu):
    VERSION = 0
    DATASET_PATH = dataset_dir + 'virology'
    DATASET_NAME = 'virology'
