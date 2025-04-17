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

class KmerCascadeCache:
    """
    Multi-level k-mer caching system: optimized for caching DNA subsequences at different lengths
    """
    
    def __init__(self, base_tokenizer=None):
        self.tokenizer = base_tokenizer
            
        # Main cache dictionary - stores full processing results
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Get k-mer parameters from the original tokenizer
        self.k = getattr(self.tokenizer, 'k', 4)
        self.stride = getattr(self.tokenizer, 'stride', 4)
        
    # Level 1 cache - 32-base subsequences (8 k-mers)
    @functools.lru_cache(maxsize=16384)
    def _tokenize_32bp(self, seq):
        """Cache tokenization results for 32bp (8 k-mers) subsequences"""
        # Only process exactly 32bp long sequences
        if len(seq) == 32:
            return self.tokenizer.tokenize(seq)
        return None
    
    # Level 2 cache - 16-base subsequences (4 k-mers)
    @functools.lru_cache(maxsize=1024)
    def _tokenize_16bp(self, seq):
        """Cache tokenization results for 16bp (4 k-mers) subsequences"""
        if len(seq) == 16:
            return self.tokenizer.tokenize(seq)
        return None
    
    # Level 3 cache - 8-base subsequences (2 k-mers)
    @functools.lru_cache(maxsize=1024)
    def _tokenize_8bp(self, seq):
        """Cache tokenization results for 8bp (2 k-mers) subsequences"""
        if len(seq) == 8:
            return self.tokenizer.tokenize(seq)
        return None
    
    # Level 4 cache - individual k-mers (4 bases)
    @functools.lru_cache(maxsize=256)
    def _tokenize_kmer(self, seq):
        """Cache individual k-mer (4bp)"""
        if len(seq) == 4:
            return [seq]  # Return single k-mer directly
        return None
    
    def _tokenize_with_cascade(self, seq):
        """Tokenize sequence using multi-level cache strategy"""
        if not seq:
            return []
            
        result = []
        seq_len = len(seq)
        
        # Process sequence using different cache levels, prioritizing larger chunks
        pos = 0
        while pos < seq_len - self.k + 1:
            remaining = seq_len - pos
            
            # Try using 32bp cache
            if remaining >= 32:
                chunk = seq[pos:pos+32]
                tokens = self._tokenize_32bp(chunk)
                if tokens:
                    result.extend(tokens)
                    pos += 32
                    continue
                    
            # Try using 16bp cache
            if remaining >= 16:
                chunk = seq[pos:pos+16]
                tokens = self._tokenize_16bp(chunk)
                if tokens:
                    result.extend(tokens)
                    pos += 16
                    continue
            
            # Try using 8bp cache
            if remaining >= 8:
                chunk = seq[pos:pos+8]
                tokens = self._tokenize_8bp(chunk)
                if tokens:
                    result.extend(tokens)
                    pos += 8
                    continue
            
            # Try using single k-mer cache
            if remaining >= 4:
                chunk = seq[pos:pos+4]
                tokens = self._tokenize_kmer(chunk)
                if tokens:
                    result.extend(tokens)
                    pos += 4
                    continue
                    
            # If all cache levels miss, process a single k-mer with the base method
            kmer = seq[pos:pos+self.k]
            if len(kmer) == self.k:
                result.append(kmer)
            pos += self.stride
                
        return result
    
    def __call__(self, text, padding="max_length", truncation=True, max_length=660, return_tensors="pt"):
        """Support the same interface as the original tokenizer"""
        # Create cache key based on parameters
        cache_key = (text, padding, truncation, max_length, return_tensors)
        
        # Check top-level cache
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # Cache miss, use multi-level caching strategy for tokenization
        self.cache_misses += 1
        
        # For short sequences, use original tokenizer directly
        if len(text) <= 32:
            result = self.tokenizer(
                text, 
                padding=padding, 
                truncation=truncation, 
                max_length=max_length, 
                return_tensors=return_tensors
            )
        else:
            # Use custom tokenization method with cascade caching
            # Simplified - in practice, would need more complex processing
            # to handle padding and special tokens correctly
            tokens = self._tokenize_with_cascade(text)
            
            # Fall back to original tokenizer for final processing
            # This could be further optimized in the future
            result = self.tokenizer(
                text, 
                padding=padding, 
                truncation=truncation, 
                max_length=max_length, 
                return_tensors=return_tensors
            )
        
        # Store result in cache
        self.cache[cache_key] = result
        return result
    
    def get_cache_stats(self):
        """Return cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        # Get stats from each cache level
        l1_info = self._tokenize_32bp.cache_info()
        l2_info = self._tokenize_16bp.cache_info()
        l3_info = self._tokenize_8bp.cache_info()
        l4_info = self._tokenize_kmer.cache_info()
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "level1_cache": {
                "hits": l1_info.hits,
                "misses": l1_info.misses,
                "size": l1_info.currsize,
                "maxsize": l1_info.maxsize
            },
            "level2_cache": {
                "hits": l2_info.hits,
                "misses": l2_info.misses,
                "size": l2_info.currsize,
                "maxsize": l2_info.maxsize
            },
            "level3_cache": {
                "hits": l3_info.hits,
                "misses": l3_info.misses, 
                "size": l3_info.currsize,
                "maxsize": l3_info.maxsize
            },
            "level4_cache": {
                "hits": l4_info.hits,
                "misses": l4_info.misses,
                "size": l4_info.currsize,
                "maxsize": l4_info.maxsize
            }
        }
    
class BatchKmerTokenizer:
    """
    True batch processing implementation for KmerTokenizer that directly handles
    multiple sequences in a single operation for improved performance
    """
    
    def __init__(self, base_tokenizer=None, batch_size=128):
        self.tokenizer = base_tokenizer
        self.batch_size = batch_size
        
        # Extract parameters from base tokenizer
        self.k = base_tokenizer.k
        self.stride = base_tokenizer.stride
        self.max_len = base_tokenizer.max_len
        self.vocab_dict = base_tokenizer.vocab_dict
        self.unk_token_id = self.vocab_dict.get("[UNK]", 1)
        
        # Stats counters
        self.process_count = 0
        self.batch_count = 0
    
    def batch_tokenize(self, texts, padding=False):
        """Tokenize multiple sequences at once without individual calls"""
        batch_tokens = []
        
        for text in texts:
            # Apply original tokenizer's length truncation/padding logic
            if len(text) > self.max_len:
                text = text[:self.max_len]
            if padding:
                if len(text) < self.max_len:
                    text = text + 'N' * (self.max_len - len(text))
            
            # Extract k-mers directly using the stride parameter
            tokens = [text[i:i + self.k] for i in range(0, len(text) - self.k + 1, self.stride)]
            batch_tokens.append(tokens)
            
        return batch_tokens
    
    def batch_convert_tokens_to_ids(self, batch_tokens):
        """Convert batches of tokens to IDs in a vectorized operation"""
        batch_ids = []
        unk_id = self.unk_token_id
        
        for tokens in batch_tokens:
            # Convert tokens to IDs using vocabulary lookup
            ids = [self.vocab_dict.get(token, unk_id) for token in tokens]
            batch_ids.append(ids)
            
        return batch_ids
    
    def __call__(self, texts, padding=False, truncation=True, max_length=660, return_tensors="pt"):
        """Efficiently process both single inputs and batches"""
        import torch
        
        # Handle single text input
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        
        # Update stats
        self.process_count += len(texts)
        self.batch_count += 1
        
        # Process entire batch at once
        batch_tokens = self.batch_tokenize(texts, padding=padding)
        batch_ids = self.batch_convert_tokens_to_ids(batch_tokens)
        
        # Create attention masks and token type IDs
        batch_attention_masks = []
        batch_token_type_ids = []
        
        for ids in batch_ids:
            # Create attention mask (1 for all tokens by default, same as original)
            attention_mask = [1 for _ in ids]
            # Create token type IDs (all zeros)
            token_type_ids = [0] * len(ids)
            
            batch_attention_masks.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
        
        # Convert to tensor format if requested
        if return_tensors == "pt":
            # Handle variable sequence lengths with padding
            if padding == "max_length":
                # Determine max length for padding within batch
                max_len = max_length
                
                # Pad all sequences to max_length
                padded_ids = []
                padded_attention_masks = []
                padded_token_type_ids = []
                
                for ids, mask, type_ids in zip(batch_ids, batch_attention_masks, batch_token_type_ids):
                    # Truncate if needed
                    if truncation and len(ids) > max_len:
                        ids = ids[:max_len]
                        mask = mask[:max_len]
                        type_ids = type_ids[:max_len]
                    
                    # Pad with zeros
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
            
            # Convert to tensors
            batch_ids = torch.tensor(batch_ids)
            batch_attention_masks = torch.tensor(batch_attention_masks)
            batch_token_type_ids = torch.tensor(batch_token_type_ids)
        
        # Create output dictionary
        result = {
            "input_ids": batch_ids,
            "attention_mask": batch_attention_masks,
            "token_type_ids": batch_token_type_ids
        }
        
        # Return single result or batch based on input type
        if single_input and return_tensors == "pt":
            return {
                "input_ids": result["input_ids"][0].unsqueeze(0),
                "attention_mask": result["attention_mask"][0].unsqueeze(0),
                "token_type_ids": result["token_type_ids"][0].unsqueeze(0)
            }
        
        return result
    
    def process_large_dataset(self, dna_input, padding="max_length", truncation=True, max_length=660, return_tensors="pt"):
        """Process large datasets in batches but keep individual results"""
        import torch
        from tqdm import tqdm
        
        all_tokenized_sequences = []
        
        # Process in batches
        for i in tqdm(range(0, len(dna_input), self.batch_size), desc="Tokenizing DNA sequences in batches"):
            batch = dna_input[i:i+self.batch_size]
            
            # Use the batch processing capability
            batch_results = self(
                batch, 
                padding=padding, 
                truncation=truncation, 
                max_length=max_length, 
                return_tensors=return_tensors
            )
            
            # Extract individual sequences for return
            if return_tensors == "pt":
                for j in range(len(batch)):
                    # Create a single sequence tensor with batch dimension
                    sequence_tensor = batch_results["input_ids"][j].unsqueeze(0)
                    all_tokenized_sequences.append(sequence_tensor)
            else:
                # For non-tensor returns, keep the list format
                for j in range(len(batch)):
                    all_tokenized_sequences.append([batch_results["input_ids"][j]])
        
        return all_tokenized_sequences
    
    def get_cache_stats(self):
        """Compatibility method for cache stats"""
        return {
            "batch_processing": {
                "sequences_processed": self.process_count,
                "batches_processed": self.batch_count,
                "batch_size": self.batch_size
            }
        }

class BatchCascadeTokenizer:
    """
    结合了 BatchKmerTokenizer 批处理能力和 KmerCascadeCache 多级缓存优势的高效实现
    """
    
    def __init__(self, base_tokenizer=None, batch_size=128):
        self.tokenizer = base_tokenizer
        self.batch_size = batch_size
        
        # 从基础分词器提取参数
        self.k = base_tokenizer.k
        self.stride = base_tokenizer.stride
        self.max_len = base_tokenizer.max_len
        self.vocab_dict = base_tokenizer.vocab_dict
        self.unk_token_id = self.vocab_dict.get("[UNK]", 1)
        
        # 统计计数
        self.process_count = 0
        self.batch_count = 0
        
        # 主缓存字典 - 存储完整处理结果
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 批处理结果缓存 - 为常见的批处理操作缓存结果
        self.batch_cache = {}
        self.batch_cache_hits = 0
        self.batch_cache_misses = 0
        
    # 子序列缓存 - 使用函数装饰器进行自动缓存
    @functools.lru_cache(maxsize=16384)
    def _tokenize_subsequence(self, seq):
        """缓存子序列的分词结果"""
        if len(seq) < self.k:
            return []
            
        # 直接提取 k-mers
        tokens = [seq[i:i + self.k] for i in range(0, len(seq) - self.k + 1, self.stride)]
        
        # 转换为 ID
        ids = [self.vocab_dict.get(token, self.unk_token_id) for token in tokens]
        
        return ids
    
    def _create_attention_mask_and_token_types(self, ids):
        """为 ID 创建注意力掩码和token类型ID"""
        attention_mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
        return attention_mask, token_type_ids
    
    def batch_tokenize(self, texts, padding=False):
        """一次性分词多个序列，利用缓存加速处理"""
        batch_tokens = []
        batch_ids = []
        batch_attention_masks = []
        batch_token_type_ids = []
        
        # 处理批次中的每个序列
        for text in texts:
            # 应用原始分词器的长度截断/填充逻辑
            if len(text) > self.max_len:
                text = text[:self.max_len]
            if padding:
                if len(text) < self.max_len:
                    text = text + 'N' * (self.max_len - len(text))
            
            # 创建缓存键
            cache_key = (text, padding)
            
            # 检查缓存
            if cache_key in self.cache:
                self.cache_hits += 1
                cached_result = self.cache[cache_key]
                batch_ids.append(cached_result["ids"])
                batch_attention_masks.append(cached_result["attention_mask"])
                batch_token_type_ids.append(cached_result["token_type_ids"])
                continue
                
            # 缓存未命中，使用缓存子序列处理
            self.cache_misses += 1
            
            # 短序列直接处理
            if len(text) <= 32:
                tokens = [text[i:i + self.k] for i in range(0, len(text) - self.k + 1, self.stride)]
                ids = [self.vocab_dict.get(token, self.unk_token_id) for token in tokens]
            else:
                # 分割序列并处理各部分
                ids = []
                pos = 0
                
                while pos < len(text) - self.k + 1:
                    # 尝试使用较大的缓存块
                    chunk_size = min(32, len(text) - pos)
                    chunk = text[pos:pos+chunk_size]
                    
                    # 利用缓存处理子序列
                    chunk_ids = self._tokenize_subsequence(chunk)
                    ids.extend(chunk_ids)
                    
                    pos += chunk_size
            
            # 创建辅助数据
            attention_mask, token_type_ids = self._create_attention_mask_and_token_types(ids)
            
            # 添加到批处理结果
            batch_ids.append(ids)
            batch_attention_masks.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            
            # 存入缓存
            self.cache[cache_key] = {
                "ids": ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            
            # 限制缓存大小
            if len(self.cache) > 100000:
                # 简单策略：删除最旧的20%条目
                keys_to_remove = list(self.cache.keys())[:20000]
                for k in keys_to_remove:
                    del self.cache[k]
            
        return batch_ids, batch_attention_masks, batch_token_type_ids
    
    def __call__(self, texts, padding=False, truncation=True, max_length=660, return_tensors="pt"):
        """支持与原始分词器相同的接口，但具有批处理和缓存能力"""
        import torch
        
        # 处理单个文本输入
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        
        # 更新统计信息
        self.process_count += len(texts)
        self.batch_count += 1
        
        # 检查批处理缓存
        cache_key = (tuple(texts), padding, truncation, max_length, return_tensors)
        if len(texts) <= 10 and cache_key in self.batch_cache:  # 仅为小批次缓存完整结果
            self.batch_cache_hits += 1
            return self.batch_cache[cache_key]
        
        self.batch_cache_misses += 1
        
        # 一次性处理整个批次
        batch_ids, batch_attention_masks, batch_token_type_ids = self.batch_tokenize(texts, padding=padding)
        
        # 转换为张量格式（如果需要）
        if return_tensors == "pt":
            # 处理可变序列长度的填充
            if padding == "max_length":
                # 确定用于填充的最大长度
                max_len = max_length
                
                # 将所有序列填充到 max_length
                padded_ids = []
                padded_attention_masks = []
                padded_token_type_ids = []
                
                for ids, mask, type_ids in zip(batch_ids, batch_attention_masks, batch_token_type_ids):
                    # 截断（如果需要）
                    if truncation and len(ids) > max_len:
                        ids = ids[:max_len]
                        mask = mask[:max_len]
                        type_ids = type_ids[:max_len]
                    
                    # 使用零填充
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
            
            # 转换为张量
            batch_ids = torch.tensor(batch_ids)
            batch_attention_masks = torch.tensor(batch_attention_masks)
            batch_token_type_ids = torch.tensor(batch_token_type_ids)
        
        # 创建输出字典
        result = {
            "input_ids": batch_ids,
            "attention_mask": batch_attention_masks,
            "token_type_ids": batch_token_type_ids
        }
        
        # 如果是小批次，缓存结果
        if len(texts) <= 10:
            self.batch_cache[cache_key] = result
            
            # 限制批处理缓存大小
            if len(self.batch_cache) > 1000:
                # 删除最旧的条目
                keys_to_remove = list(self.batch_cache.keys())[:200]
                for k in keys_to_remove:
                    del self.batch_cache[k]
        
        # 根据输入类型返回单个结果或批次
        if single_input and return_tensors == "pt":
            return {
                "input_ids": result["input_ids"][0].unsqueeze(0),
                "attention_mask": result["attention_mask"][0].unsqueeze(0),
                "token_type_ids": result["token_type_ids"][0].unsqueeze(0)
            }
        
        return result
    
    def process_large_dataset(self, dna_input, padding="max_length", truncation=True, max_length=660, return_tensors="pt"):
        """处理大型数据集，按批次处理但保留单个结果"""
        import torch
        from tqdm import tqdm
        
        all_tokenized_sequences = []
        
        # 按批次处理
        for i in tqdm(range(0, len(dna_input), self.batch_size), desc="Tokenizing DNA sequences in batches (with cascade caching)"):
            batch = dna_input[i:i+self.batch_size]
            
            # 利用批处理和缓存能力
            batch_results = self(
                batch, 
                padding=padding, 
                truncation=truncation, 
                max_length=max_length, 
                return_tensors=return_tensors
            )
            
            # 提取单个序列以返回
            if return_tensors == "pt":
                for j in range(len(batch)):
                    # 创建带有batch维度的单个序列张量
                    sequence_tensor = batch_results["input_ids"][j].unsqueeze(0)
                    all_tokenized_sequences.append(sequence_tensor)
            else:
                # 对于非张量返回，保持列表格式
                for j in range(len(batch)):
                    all_tokenized_sequences.append([batch_results["input_ids"][j]])
        
        return all_tokenized_sequences
    
    def get_cache_stats(self):
        """返回缓存统计信息"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        batch_total = self.batch_cache_hits + self.batch_cache_misses
        batch_hit_rate = self.batch_cache_hits / batch_total if batch_total > 0 else 0
        
        # 获取子序列缓存的统计信息
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