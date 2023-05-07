import transformers
import torch
from lm_eval.base import BaseLM
import torch.nn as nn

class OPTLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="facebook/opt-350m",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )
              
        self.model.to(self._device)
        
        self.model.eval()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
        )


        self.vocab_size = self.tokenizer.vocab_size


        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size


    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if hasattr(self, "_devices"):
            return self.model.module.config.max_position_embeddings
        else:
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :50272]

    def _model_generate(self, context, max_length, eos_token_id):
        
        completion = self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, top_k=5, no_repeat_ngram_size=4, num_return_sequences=1
        )
                
        return completion


