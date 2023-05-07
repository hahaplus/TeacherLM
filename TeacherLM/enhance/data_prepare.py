from datasets import load_dataset, load_from_disk
from pathlib import Path
import os, jsonlines


def remove_duplicated_block(input_str):
    """
    Note that duplicated blanks will impact TeacherLM's analysis, so remove it with caution.
    """
    output_str = input_str.replace("  ", " ")
    if output_str == "":
        print("input str is empty")
        return ""
    if output_str != "" and output_str[-1] == " ":
        output_str = output_str.strip(" ")
    return output_str


def prepare_CREAK():


    def add_prompts(example):
        """
        Add prompts to your example.
        """
        answer = "A" if str(example["label"]) == "true" else "B"
        text = "Q: Determine the correctness of the following sentence. " + example["sentence"] + f" (A) True (B) False A: The answer is ({answer})."
        cot_text = text + " Let's think step by step."
        error_text = text + " The common mistakes are:"
        fundamental_text = text + " The fundamental of this question is:"
        cot_text, error_text, fundamental_text = map(remove_duplicated_block, [cot_text, error_text, fundamental_text])
        example["text"] = text
        example["cot_text"] = cot_text
        example["error_text"] = error_text
        example["fundamental_text"] = fundamental_text
        return example


    def fetech_dataset():
        """
        Fetch the dataset from huggingface's dataset.
        """
        dataset = load_dataset("amydeng2000/CREAK")
        dataset.save_to_disk("./CREAK")


    def save_to_jsonl(dataset, path):
        """
        Save the dataset to jsonl format.
        """
        if "train" in dataset:
            dataset = dataset["train"]
        with jsonlines.open(path, "w") as writer:
            for each in dataset:
                writer.write(each)


    if os.path.exists("./CREAK"):
        dataset = load_from_disk("./CREAK")
    else:
        fetech_dataset()
        dataset = load_from_disk("./CREAK")
    dataset = dataset.map(add_prompts)
    dataset.save_to_disk("./CREAK")
    save_to_jsonl(dataset, "./CREAK.jsonl")


def post_process_CREAK():
    def creat_cot_tail(example):
        example["cot_tail"] = remove_duplicated_block(example["all"])
        return example

    dataset = load_dataset("json", data_files="./CREAK_cot.jsonl").map(creat_cot_tail)
    if "train" in dataset:
        dataset = dataset["train"]
    dataset.save_to_disk("CREAK")
    return


if __name__ == "__main__":
    if not os.path.exists("./CREAK.jsonl"):
        prepare_CREAK()
    if os.path.exists("./CREAK_cot.jsonl"):
        post_process_CREAK()
    CREAK = load_from_disk("./CREAK")
    print(CREAK)