import argparse, gc, math, os, time, jsonlines, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank", required=False, type=int, help="used by dist launchers"
    )
    parser.add_argument("--name", type=str, help="Name path", required=True)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--num_tokens", default=2048, type=int, help="maximum number of tokens")
    parser.add_argument(
        "--dtype",
        type=str,
        help="float16 or int8",
        choices=["int8", "float16"],
        default="float16",
    )
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--task_type", type=str)
    return parser.parse_args()


def print_rank0(*msg):
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if rank != 0:
        return
    print(*msg)


def get_max_memory_per_gpu_dict(dtype, model_name):
    """try to generate the memory map based on what we know about the model and the available hardware"""
    n_gpus = torch.cuda.device_count()
    try:
        config = AutoConfig.from_pretrained(model_name)
        h = config.hidden_size
        l = config.n_layer
        v = config.vocab_size
        model_params = l * (12 * h ** 2 + 13 * h) + v * h + 4 * h
    except:
        print_rank0(
            f"The model {model_name} has a broken config file. Please notify the owner"
        )
        raise Exception("Broken config file")
    if dtype == torch.int8:
        bytes = 1
    else:
        bytes = torch.finfo(dtype).bits / 8
    param_memory_total_in_bytes = model_params * bytes
    # add 5% since weight sizes aren't the same and some GPU may need more memory
    param_memory_per_gpu_in_bytes = int(param_memory_total_in_bytes / n_gpus * 1.10)
    print_rank0(
        f"Estimating {param_memory_per_gpu_in_bytes/2**30:0.2f}GB per gpu for weights"
    )
    # check the real available memory
    # load cuda kernels first and only measure the real free memory after loading (shorter by ~2GB)
    torch.ones(1).cuda()
    max_memory_per_gpu_in_bytes = torch.cuda.mem_get_info(0)[0]
    if max_memory_per_gpu_in_bytes < param_memory_per_gpu_in_bytes:
        raise ValueError(
            f"Unable to generate the memory map automatically as the needed estimated memory per gpu ({param_memory_per_gpu_in_bytes/2**30:0.2f}GB) is bigger than the available per gpu memory ({max_memory_per_gpu_in_bytes/2**30:0.2f}GB)"
        )
    max_memory_per_gpu = {
        i: param_memory_per_gpu_in_bytes for i in range(torch.cuda.device_count())
    }
    print("Max memory per gpu:", max_memory_per_gpu)
    return max_memory_per_gpu


def generate(tokenizer,inputs,model,generate_kwargs):
    """returns a list of zipped inputs, outputs and number of new tokens"""
    input_tokens = tokenizer.batch_encode_plus(
        inputs, return_tensors="pt", padding=True
    )
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda:0")
    outputs = model.generate(**input_tokens, **generate_kwargs)
    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]
    total_new_tokens = [
        o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)
    ]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return zip(inputs, outputs, total_new_tokens)

def main():
    # Initialize
    args = get_args()
    num_tokens = args.num_tokens
    world_size = torch.cuda.device_count()
    print_rank0(f"Using {world_size} gpus")
    model_name = args.name
    print_rank0(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    dtype = torch.int8 if args.dtype=='int8' else torch.float16
    kwargs = dict(
        device_map="auto", max_memory=get_max_memory_per_gpu_dict(dtype, model_name),
    )
    if dtype == "int8":
        print_rank0("Using `load_in_8bit=True` to use quanitized model")
        kwargs["load_in_8bit"] = True
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Generate
    print_rank0(
        f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}"
    )
    input_sentences = []
    doc_ = []
    task_type = args.task_type
    print_rank0("current using task_type is: " + task_type)
    output_type_dict = {
        "error_text": "error",
        "fundamental_text": "fundamental",
        "cot_text": "cot",
    }
    output_type = output_type_dict[task_type]
    print_rank0("current writing output_type is: " + output_type)
    with jsonlines.open(args.input, "r") as f:
        for doc in f:
            doc_.append(doc)
            input_sentences.append(f"{doc[task_type]}")
    if args.batch_size > len(input_sentences):
        # dynamically extend to support larger bs by repetition
        input_sentences *= math.ceil(args.batch_size / len(input_sentences))
    generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=True)
    print_rank0(f"Generate args {generate_kwargs}")
    print_rank0(f"*** Running generate")
    bs_num = len(input_sentences) // args.batch_size
    j = 0
    with jsonlines.open(args.output, "w") as f:
        for i in tqdm(range(bs_num + 1)):
            if i == bs_num and len(input_sentences) > bs_num * args.batch_size:
                inputs = input_sentences[i * args.batch_size :]
            elif i == bs_num and len(input_sentences) <= bs_num * args.batch_size:
                break
            else:
                inputs = input_sentences[
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            generated = generate(tokenizer,inputs,model,generate_kwargs)
            for i, o, _ in generated:
                if "Let's think step by step." in i:
                    index_cot = o.index("Let's think step by step.")
                    doc_[j][output_type] = o[
                        index_cot + len("Let's think step by step.") :
                    ]
                elif "The fundamental of this question is:" in i:
                    index_fun = o.index("The fundamental of this question is:")
                    doc_[j][output_type] = o[
                        index_fun + len("The fundamental of this question is:") :
                    ]
                elif "The common mistakes are:" in i:
                    index_err = o.index("The common mistakes are:")
                    doc_[j][output_type] = o[
                        index_err + len("The common mistakes are:") :
                    ]
                doc_[j]["all"] = o
                if int(os.getenv("LOCAL_RANK", "0")) == 0:
                    f.write(doc_[j])
                    j += 1

if __name__ == "__main__":
    main()   
