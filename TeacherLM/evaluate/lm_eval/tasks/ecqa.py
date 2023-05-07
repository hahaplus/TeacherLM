import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
import json
import re
import datasets

dataset_dir = 'lm_eval/datasets/'

class ECQA(Task):
    VERSION = 0
    DATASET_PATH = dataset_dir + "ecqa"
    DATASET_NAME = "ecqa"
    
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = datasets.Dataset.load_from_disk(self.DATASET_PATH)
        
    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset
    
    def doc_to_text(self, doc):
        ret_ctx = f"Q: {doc['question']} A: The answer is ("
        return ret_ctx
    
    def construct_requests(self, doc, ctx):
        ll_A, _ = rf.loglikelihood(ctx, "A")
        ll_B, _ = rf.loglikelihood(ctx, "B")
        ll_C, _ = rf.loglikelihood(ctx, "C")
        ll_D, _ = rf.loglikelihood(ctx, "D")
        ll_E, _ = rf.loglikelihood(ctx, "E")
        requests = [ll_A, ll_B, ll_C, ll_D, ll_E]
        return requests
    
    def doc_to_target(self, doc):
        return " "
    
    def process_results(self, doc, results):
        gold = doc["answer"]
        option_term=['(A)','(B)','(C)','(D)','(E)','(F)','(G)','(H)']
        pred = np.argmax(results)
        acc = 1.0 if option_term[pred] == gold else 0.0
        
        return {"acc": acc}
    
    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
    
