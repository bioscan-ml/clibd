from transformers import AutoTokenizer, BertModel, logging
import torch
import torch.nn as nn
import math
from torch import Tensor
import warnings
from bioscanclip.util.util import create_child_from_parent
import clip
from loratorch.layers import MultiheadAttention as LoRA_MultiheadAttention
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pre_trained_bert(language_model_name):
    logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(language_model_name)
    model = BertModel.from_pretrained(language_model_name)
    logging.set_verbosity_info()
    for param in model.parameters():
        param.requires_grad = False

    return tokenizer, model

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


class CLIBDLanguageEncoder(nn.Module):
    def __init__(self, model, r: int, num_classes: int = 0, lora_layer=None):
        super(CLIBDLanguageEncoder, self).__init__()

        assert r > 0
        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(model.encoder.layer)))

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in model.parameters():
            param.requires_grad = False

        for layer_idx, layer in enumerate(model.encoder.layer):
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
        self.base_language_encoder = model

        if num_classes > 0:
            self.proj = nn.Linear(self.base_language_encoder.pooler.dense.out_features, num_classes)


    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x) -> Tensor:

        return self.proj(self.base_language_encoder(**x).last_hidden_state.mean(dim=1))

class LoRA_bert_OpenCLIP(nn.Module):
    def __init__(self, bert_model, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_bert_OpenCLIP, self).__init__()
        if num_classes != 768:
            raise ValueError(
                "num_classes should be 768 for OpenCLIP, may need to implement a new head for other num_classes")
        assert r > 0
        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(bert_model.resblocks)))
            block_list = enumerate(bert_model.resblocks)

        # lets freeze first
        for param in bert_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in block_list:
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            blk.attn = create_child_from_parent(blk.attn, LoRA_MultiheadAttention, embed_dim=blk.attn.embed_dim, num_heads=blk.attn.num_heads, enable_lora=['q', 'k', 'v'], r=r)

        self.lora_bert = bert_model

    def forward(self, x: Tensor) -> Tensor:
        x = clip.tokenize(x).to(DEVICE)
        print(x)
        print(self.lora_bert)

        return self.lora_bert(x)
