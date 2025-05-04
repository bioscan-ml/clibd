import math
from itertools import product

import torch
import torch.nn as nn
from torch import Tensor
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertConfig, BertForMaskedLM
from bioscanclip.util.util import remove_extra_pre_fix


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_pre_trained_bioscan_bert(bioscan_bert_checkpoint, k=5):
    """Load pre-trained BioScan BERT model with k-mer vocabulary
    
    Args:
        bioscan_bert_checkpoint: Path to checkpoint file
        k: k-mer size (default: 5)
    """
    
    # Build k-mer vocabulary
    kmer_iter = (["".join(kmer)] for kmer in product("ACGT", repeat=k))
    vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])
    vocab_size = len(vocab)

    # Load checkpoint
    ckpt = torch.load(bioscan_bert_checkpoint, map_location=device)
    model_ckpt = remove_extra_pre_fix(ckpt["model"]) if "model" in ckpt else ckpt

    # Configure and initialize model
    config_dict = ckpt.get("bert_config", {"vocab_size": vocab_size, "output_hidden_states": True})
    bert_config = BertConfig(**config_dict)
    model = BertForMaskedLM(bert_config)

    # Clean up checkpoint
    if "bert.embeddings.position_ids" in model_ckpt:
        model_ckpt.pop("bert.embeddings.position_ids")

    for key in ["classifier.weight", "classifier.bias"]:
        if key in model_ckpt:
            print(f"Removing unexpected key: {key}")
            model_ckpt.pop(key)

    # Load state dict
    model.load_state_dict(model_ckpt, strict=False)
    return model.to(device)

class KmerTokenizerWithAttMask(object):
    def __init__(self, k: int = 5, max_len: int = 660, stride: int = None):
        """
        A tokenizer that:
         1) pads/truncates DNA to max_len with 'N'
         2) splits into k-mers (stride defaults to k for non-overlap)
         3) builds a vocab over all ACGT k-mers + specials
         4) produces input_ids, attention_mask, token_type_ids
        """
        self.k = k
        self.max_len = max_len
        self.stride = stride or k

        if stride is None:
            self.stride = k
        else:
            self.stride = stride

        # build vocab once
        # kmer_iter = ("".join(kmer) for kmer in product("ACGT", repeat=k))
        # specials = ["<MASK>", "<CLS>", "<UNK>"]
        # vocab = build_vocab_from_iterator(kmer_iter, specials=specials)
        # vocab.set_default_index(vocab["<UNK>"])

        kmer_iter = (["".join(kmer)] for kmer in product("ACGT", repeat=k))
        vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
        vocab.set_default_index(vocab["<UNK>"])

        self.vocab = vocab

    def __call__(self, dna_sequence: str):
        # 1) pad or truncate to max_len
        if len(dna_sequence) >= self.max_len:
            seq = dna_sequence[: self.max_len]
        else:
            seq = dna_sequence + "N" * (self.max_len - len(dna_sequence))

        # 2) split into k-mers
        tokens = [
            seq[i : i + self.k]
            for i in range(0, len(seq) - self.k + 1, self.stride)
        ]


        # 3) attention mask: 1 for any k-mer containing A/C/G/T, 0 if all 'N'
        attention_mask = [
            0 if any(base == "N" for base in token) else 1
            for token in tokens
        ]
        # 4) convert tokens to IDs
        input_ids = [self.vocab[token] for token in tokens]
        input_ids = [0, *input_ids]  # add CLS token at the beginning
        # 4.5) Fix attention mask
        attention_mask = [1, *attention_mask]

        # 5) single-segment token types (all zeros)
        token_type_ids = [0] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }

# Old kmer tokenizer
class KmerTokenizer(object):
    def __init__(self, k, stride=1):
        self.k = k
        self.stride = stride

    def __call__(self, dna_sequence):
        tokens = []
        for i in range(0, len(dna_sequence) - self.k + 1, self.stride):
            k_mer = dna_sequence[i : i + self.k]
            tokens.append(k_mer)
        return tokens

class PadSequence(object):
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, dna_sequence):
        if len(dna_sequence) > self.max_len:
            return dna_sequence[: self.max_len]
        else:
            return dna_sequence + "N" * (self.max_len - len(dna_sequence))

def get_sequence_pipeline(k=5):
    kmer_iter = (["".join(kmer)] for kmer in product("ACGT", repeat=k))
    vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])

    max_len = 660
    pad = PadSequence(max_len)
    tokenizer = KmerTokenizer(k, stride=k)
    sequence_pipeline = lambda x: [0, *vocab(tokenizer(pad(x)))]

    return sequence_pipeline


# MODIFIED FROM https://github.com/JamesQFreeman/LoRA-barcode_bert/blob/main/lora.py

class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class CLIBDDNAEncoder(nn.Module):
    def __init__(self, model, r: int, num_classes: int = 0, lora_layer=None):
        super(CLIBDDNAEncoder, self).__init__()

        assert r > 0
        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(model.bert.encoder.layer)))

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in model.parameters():
            param.requires_grad = False

        for layer_idx, layer in enumerate(model.bert.encoder.layer):
            if layer_idx not in self.lora_layer:
                continue
            w_q_linear = layer.attention.self.query
            w_v_linear = layer.attention.self.value
            dim = layer.attention.self.query.in_features

            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            layer.attention.self.query = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            layer.attention.self.value = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.base_dna_encoder = model

        if num_classes > 0:
            self.base_dna_encoder.cls.predictions.decoder = nn.Linear(
                self.base_dna_encoder.cls.predictions.decoder.in_features, num_classes)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, input) -> Tensor:
        input_ids = input["input_ids"].to(device)
        attention_mask = input["attention_mask"].to(device)
        token_type_ids = input["token_type_ids"].to(device)
        outputs = self.base_dna_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[-1]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            masked_h = hidden_states * mask
            sum_h = masked_h.sum(dim=1)
            lengths = mask.sum(dim=1)
            mean_h = sum_h / lengths
            return mean_h
        else:
            return hidden_states.mean(dim=1)

class Freeze_DNA_Encoder(nn.Module):
    def __init__(self):
        super(Freeze_DNA_Encoder, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x
