import math
from itertools import product

import functools
import torch
import torch.nn as nn
from torch import Tensor
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertConfig, BertForMaskedLM
from bioscanclip.util.util import PadSequence, KmerTokenizer, load_bert_model, remove_extra_pre_fix
import concurrent.futures
from multiprocessing import cpu_count

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_pre_trained_bioscan_bert(bioscan_bert_checkpoint, k=5):
    """Load pre-trained BioScan BERT model with k-mer vocabulary
    
    Args:
        bioscan_bert_checkpoint: Path to checkpoint file
        k: k-mer size (default: 5)
    """
    print(f"\nLoading model from {bioscan_bert_checkpoint}")
    
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

    def forward(self, sequence) -> Tensor:
        """
        TODO: change to "return self.base_dna_encoder(x).hidden_states[-1].mean(dim=1)"
        TODO: Then also retrain the models.
        """

        return self.base_dna_encoder(sequence).logits.softmax(dim=-1).mean(dim=1)

class Freeze_DNA_Encoder(nn.Module):
    def __init__(self):
        super(Freeze_DNA_Encoder, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x

    """
    """
    
    def __init__(self, base_tokenizer=None, batch_size=128):
        self.tokenizer = base_tokenizer
        self.batch_size = batch_size
        
        self.k = base_tokenizer.k
        self.stride = base_tokenizer.stride
        self.max_len = base_tokenizer.max_len
        self.vocab_dict = base_tokenizer.vocab_dict
        self.unk_token_id = self.vocab_dict.get("[UNK]", 1)
        
        self.process_count = 0
        self.batch_count = 0
        
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.batch_cache = {}
        self.batch_cache_hits = 0
        self.batch_cache_misses = 0
        
    @functools.lru_cache(maxsize=16384)
    def _tokenize_subsequence(self, seq):
        if len(seq) < self.k:
            return []
            
        tokens = [seq[i:i + self.k] for i in range(0, len(seq) - self.k + 1, self.stride)]
        
        ids = [self.vocab_dict.get(token, self.unk_token_id) for token in tokens]
        
        return ids
    
    def _create_attention_mask_and_token_types(self, ids):
        attention_mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
        return attention_mask, token_type_ids
    
    def batch_tokenize(self, texts, padding=False):
        batch_tokens = []
        batch_ids = []
        batch_attention_masks = []
        batch_token_type_ids = []
        
        for text in texts:
            if len(text) > self.max_len:
                text = text[:self.max_len]
            if padding:
                if len(text) < self.max_len:
                    text = text + 'N' * (self.max_len - len(text))
            
            cache_key = (text, padding)
            
            if cache_key in self.cache:
                self.cache_hits += 1
                cached_result = self.cache[cache_key]
                batch_ids.append(cached_result["ids"])
                batch_attention_masks.append(cached_result["attention_mask"])
                batch_token_type_ids.append(cached_result["token_type_ids"])
                continue
                
            self.cache_misses += 1
            
            if len(text) <= 32:
                tokens = [text[i:i + self.k] for i in range(0, len(text) - self.k + 1, self.stride)]
                ids = [self.vocab_dict.get(token, self.unk_token_id) for token in tokens]
            else:
                ids = []
                pos = 0
                
                while pos < len(text) - self.k + 1:
                    chunk_size = min(32, len(text) - pos)
                    chunk = text[pos:pos+chunk_size]
                    
                    chunk_ids = self._tokenize_subsequence(chunk)
                    ids.extend(chunk_ids)
                    
                    pos += chunk_size
            
            attention_mask, token_type_ids = self._create_attention_mask_and_token_types(ids)
            
            batch_ids.append(ids)
            batch_attention_masks.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            
            self.cache[cache_key] = {
                "ids": ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            
            if len(self.cache) > 100000:
                keys_to_remove = list(self.cache.keys())[:20000]
                for k in keys_to_remove:
                    del self.cache[k]
            
        return batch_ids, batch_attention_masks, batch_token_type_ids
    
    def __call__(self, texts, padding=False, truncation=True, max_length=660, return_tensors="pt"):
        import torch
        
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        
        self.process_count += len(texts)
        self.batch_count += 1
        
        cache_key = (tuple(texts), padding, truncation, max_length, return_tensors)
        if len(texts) <= 10 and cache_key in self.batch_cache:  
            self.batch_cache_hits += 1
            return self.batch_cache[cache_key]
        
        self.batch_cache_misses += 1
        
        batch_ids, batch_attention_masks, batch_token_type_ids = self.batch_tokenize(texts, padding=padding)
        
        if return_tensors == "pt":
            if padding == "max_length":
                max_len = max_length
                
                padded_ids = []
                padded_attention_masks = []
                padded_token_type_ids = []
                
                for ids, mask, type_ids in zip(batch_ids, batch_attention_masks, batch_token_type_ids):
                    if truncation and len(ids) > max_len:
                        ids = ids[:max_len]
                        mask = mask[:max_len]
                        type_ids = type_ids[:max_len]
                    
                    padding_length = max_len - len(ids)
                    if padding_length > 0:
                        ids = ids + [0] * padding_length
                        mask = mask + [0] * padding_length
                        type_ids = type_ids + [0] * padding_length
                    
                    padded_ids.append(ids)
                    padded_attention_masks.append(mask)
                    padded_token_type_ids.append(type_ids)
                
                batch_ids = padded_ids
                batch_attention_masks = padded_attention_masks
                batch_token_type_ids = padded_token_type_ids
            
            batch_ids = torch.tensor(batch_ids)
            batch_attention_masks = torch.tensor(batch_attention_masks)
            batch_token_type_ids = torch.tensor(batch_token_type_ids)
        
        result = {
            "input_ids": batch_ids,
            "attention_mask": batch_attention_masks,
            "token_type_ids": batch_token_type_ids
        }
        
        if len(texts) <= 10:
            self.batch_cache[cache_key] = result
            
            if len(self.batch_cache) > 1000:
                keys_to_remove = list(self.batch_cache.keys())[:200]
                for k in keys_to_remove:
                    del self.batch_cache[k]
        
        if single_input and return_tensors == "pt":
            return {
                "input_ids": result["input_ids"][0].unsqueeze(0),
                "attention_mask": result["attention_mask"][0].unsqueeze(0),
                "token_type_ids": result["token_type_ids"][0].unsqueeze(0)
            }
        
        return result
    
    def process_large_dataset(self, dna_input, padding="max_length", truncation=True, max_length=660, return_tensors="pt"):
        import torch
        from tqdm import tqdm
        
        all_tokenized_sequences = []
        
        for i in tqdm(range(0, len(dna_input), self.batch_size), desc="Tokenizing DNA sequences in batches (with cascade caching)"):
            batch = dna_input[i:i+self.batch_size]
            
            batch_results = self(
                batch, 
                padding=padding, 
                truncation=truncation, 
                max_length=max_length, 
                return_tensors=return_tensors
            )
            
            if return_tensors == "pt":
                for j in range(len(batch)):
                    sequence_tensor = batch_results["input_ids"][j].unsqueeze(0)
                    all_tokenized_sequences.append(sequence_tensor)
            else:
                for j in range(len(batch)):
                    all_tokenized_sequences.append([batch_results["input_ids"][j]])
        
        return all_tokenized_sequences
    
    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        batch_total = self.batch_cache_hits + self.batch_cache_misses
        batch_hit_rate = self.batch_cache_hits / batch_total if batch_total > 0 else 0
        
        subsequence_info = self._tokenize_subsequence.cache_info()
        
        return {
            "process_stats": {
                "sequences_processed": self.process_count,
                "batches_processed": self.batch_count,
                "batch_size": self.batch_size
            },
            "sequence_cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": hit_rate,
                "cache_size": len(self.cache)
            },
            "batch_cache": {
                "hits": self.batch_cache_hits,
                "misses": self.batch_cache_misses,
                "hit_rate": batch_hit_rate,
                "cache_size": len(self.batch_cache)
            },
            "subsequence_cache": {
                "hits": subsequence_info.hits,
                "misses": subsequence_info.misses,
                "size": subsequence_info.currsize,
                "maxsize": subsequence_info.maxsize
            }
        }