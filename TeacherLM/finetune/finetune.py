import os, torch
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    HfArgumentParser,
    set_seed,
    default_data_collator,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint, IntervalStrategy
from datasets import Dataset, load_from_disk
from prompt_args import ModelArguments, DataArguments
from transformers.file_utils import torch_required

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable warning


@torch_required
def print_0(*args, **kwargs):
    if torch.distributed.get_rank() in (-1, 0):
        print(*args, **kwargs)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            print_0(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    train_dataset = load_from_disk(data_args.dataset_name)
    if "train" in train_dataset:
        train_dataset = train_dataset["train"]

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        use_cache=False,  # incompatible with gradient checkpointing
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if training_args.fp16:
        model.half()
    elif training_args.bf16:
        model.bfloat16()

    model = model.to(torch.cuda.current_device())
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_task(task_type, example):
        text = example[task_type]
        model_inputs = tokenizer(
            text,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )
        model_inputs_backup = tokenizer(text)
        if len(model_inputs["input_ids"]) < len(model_inputs_backup["input_ids"]):
            num = len(model_inputs_backup["input_ids"])
            print_0(f"warning: input id length exceeded ! the actual length is {num}")
            model_inputs["input_ids"] = model_inputs_backup["input_ids"][
                -data_args.max_source_length :
            ]
        model_inputs["labels"] = model_inputs["input_ids"]
        return model_inputs

    class TrainDataset(torch.utils.data.IterableDataset):
        def __init__(self, dataset, max_num_examples=None):
            super(TrainDataset, self).__init__()
            self.dataset = dataset
            self.num_examples = len(dataset)

        def __iter__(self):
            task_type = data_args.task_type
            preprocess_function = partial(preprocess_task, str(task_type))
            return map(preprocess_function, self.dataset)

        def __len__(self):
            return self.num_examples

    if data_args.max_train_samples is not None:
        train_dataset = Dataset.from_dict(train_dataset[: data_args.max_train_samples])
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = TrainDataset(train_dataset, data_args.max_train_samples)

    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(  # dynamically pad the inputs received, as well as the labels.
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
        )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    if training_args.save_strategy != IntervalStrategy.NO:
        trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None
        else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
