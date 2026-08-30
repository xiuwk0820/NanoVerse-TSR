# import gzip
# import html
# import os
# from functools import lru_cache

# import ftfy
# import regex as re
# import os
# from typing import Any, Union, List
# import packaging

# import torch

# @lru_cache()
# def default_bpe():
#     """返回预训练的BPE词汇表路径[1,3](@ref)
#     使用LRU缓存避免重复加载
#     """
#     return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model_data/bpe_simple_vocab_16e6.txt.gz")


# @lru_cache()
# def bytes_to_unicode():
#     """创建字节到Unicode字符的映射表[3,7](@ref)
#     解决BPE编码中的不可打印字符问题
#     返回：字典{字节: Unicode字符}
#     """
#     bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
#     cs = bs[:]
#     n = 0
#     # 处理不可打印字符（0-255范围内）
#     for b in range(2**8):
#         if b not in bs:
#             bs.append(b)
#             cs.append(2**8 + n)
#             n += 1
#     cs = [chr(n) for n in cs]
#     return dict(zip(bs, cs))


# def get_pairs(word):
#     """获取连续字符对集合[3,7](@ref)
#     Args:
#         word: 由多个符号组成的元组
#     Returns:
#         符号对的集合，用于BPE合并
#     """
#     pairs = set()
#     prev_char = word[0]
#     for char in word[1:]:
#         pairs.add((prev_char, char))
#         prev_char = char
#     return pairs


# def basic_clean(text):
#     """基础文本清洗：
#     1. 修复常见文本错误（ftfy）
#     2. 双重HTML转义解码[3,7](@ref)"""
#     text = ftfy.fix_text(text)
#     text = html.unescape(html.unescape(text))
#     return text.strip()


# def whitespace_clean(text):
#     """空格规范化：
#     1. 合并连续空格为单个
#     2. 去除首尾空格[3,7](@ref)"""
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()


# class SimpleTokenizer(object):
#     """基于BPE的简单分词器，支持特殊标记处理[3,7,8](@ref)"""
#     def __init__(self, bpe_path: str = default_bpe()):
#         # 初始化字节编码器/解码器
#         self.byte_encoder = bytes_to_unicode()
#         self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
#         # 加载BPE合并表（从gzip文件）
#         merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
#         merges = merges[1:49152-256-2+1]  # 保留有效合并规则
#         merges = [tuple(merge.split()) for merge in merges]
        
#         # 构建词汇表
#         vocab = list(bytes_to_unicode().values())
#         vocab += [v + '</w>' for v in vocab]  # 添加词尾标记
#         for merge in merges:
#             vocab.append(''.join(merge))
#         vocab.extend(['<|startoftext|>', '<|endoftext|>'])  # 特殊标记
        
#         # 创建编码器/解码器
#         self.encoder = dict(zip(vocab, range(len(vocab))))
#         self.decoder = {v: k for k, v in self.encoder.items()}
#         self.bpe_ranks = dict(zip(merges, range(len(merges))))
        
#         # 缓存和正则模式
#         self.cache = {'<|startoftext|>': '<|startoftext|>', 
#                      '<|endoftext|>': '<|endoftext|>'}
#         self.pat = re.compile(
#             r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|
#             [\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", 
#             re.IGNORECASE)

#     def bpe(self, token):
#         """执行BPE编码[3,7](@ref)
#         Args:
#             token: 待编码的字符串
#         Returns:
#             BPE处理后的合并结果
#         """
#         if token in self.cache:
#             return self.cache[token]
        
#         # 添加词尾标记并生成字符对
#         word = tuple(token[:-1]) + (token[-1] + '</w>',)
#         pairs = get_pairs(word)

#         if not pairs:
#             return token + '</w>'

#         # 迭代合并最高优先级的字符对
#         while True:
#             bigram = min(pairs, 
#                        key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
#             if bigram not in self.bpe_ranks:
#                 break
#             first, second = bigram
            
#             # 执行合并操作
#             new_word = []
#             i = 0
#             while i < len(word):
#                 try:
#                     j = word.index(first, i)
#                     new_word.extend(word[i:j])
#                     i = j
#                 except:
#                     new_word.extend(word[i:])
#                     break

#                 if word[i] == first and i < len(word)-1 and word[i+1] == second:
#                     new_word.append(first + second)
#                     i += 2
#                 else:
#                     new_word.append(word[i])
#                     i += 1
#             word = tuple(new_word)
#             if len(word) == 1:
#                 break
#             pairs = get_pairs(word)
        
#         # 缓存结果
#         word = ' '.join(word)
#         self.cache[token] = word
#         return word

#     def encode(self, text):
#         """文本编码主流程：
#         1. 清洗文本
#         2. 正则分词
#         3. BPE编码转换[3,7](@ref)"""
#         bpe_tokens = []
#         text = whitespace_clean(basic_clean(text)).lower()  # 标准化处理
        
#         # 分词和编码
#         for token in re.findall(self.pat, text):
#             # 字节编码转换
#             token_bytes = token.encode('utf-8')
#             token = ''.join(self.byte_encoder[b] for b in token_bytes)
#             # BPE处理
#             bpe_token = self.bpe(token).split(' ')
#             bpe_tokens.extend(self.encoder[t] for t in bpe_token)
#         return bpe_tokens

#     def decode(self, tokens):
#         """解码过程：
#         1. 通过解码器转换token到字符串
#         2. 处理词尾标记
#         3. 字节解码[3,7](@ref)"""
#         text = ''.join([self.decoder[token] for token in tokens])
#         # 处理词尾标记和字节解码
#         text = text.replace('</w>', ' ')
#         byte_data = bytearray([self.byte_decoder[c] for c in text])
#         return byte_data.decode('utf-8', errors="replace")


# def tokenize(_tokenizer, texts: Union[str, List[str]], 
#             context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
#     """将文本转换为模型输入张量[3,7](@ref)
#     Args:
#         texts: 输入文本（单条或多条）
#         context_length: 最大序列长度（CLIP标准为77）
#         truncate: 超长时是否截断
#     Returns:
#         shape: (batch_size, context_length) 的填充后张量
#     """
#     if isinstance(texts, str):
#         texts = [texts]

#     # 特殊标记编码
#     sot_token = _tokenizer.encoder["<|startoftext|>"]
#     eot_token = _tokenizer.encoder["<|endoftext|>"]
    
#     # 编码所有文本
#     all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] 
#                 for text in texts]
    
#     # 处理PyTorch版本差异
#     dtype = torch.long if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0") else torch.int
#     result = torch.zeros(len(all_tokens), context_length, dtype=dtype)
    
#     # 填充张量
#     for i, tokens in enumerate(all_tokens):
#         if len(tokens) > context_length:
#             if truncate:  # 截断处理
#                 tokens = tokens[:context_length]
#                 tokens[-1] = eot_token  # 确保结尾标记
#             else:
#                 raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
#         result[i, :len(tokens)] = torch.tensor(tokens)
    
#     return result.long()

# class EnhancedNumberTokenizer(SimpleTokenizer):
#     """增强型分词器，针对数字处理进行优化（结合网页1的tokenizer设计思想）"""
#     def __init__(self, bpe_path: str = default_bpe()):
#         super().__init__(bpe_path)
        
#         # 改进正则模式（网页1的split思想扩展）
#         self.pat = re.compile(
#             r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|
#             (?:\$|€|£|¥)?(?:[+-]?\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?%?|  # 增强数字匹配
#             [\p{L}]+|[^\s\p{L}\p{N}]+""", 
#             re.IGNORECASE | re.VERBOSE
#         )
        
#         # 数字预处理映射表
#         self.num_trans = str.maketrans('１２３４５６７８９０，', '1234567890,')
        
#     def _normalize_numbers(self, text: str) -> str:
#         """数字标准化处理（参考网页1的预处理阶段思想）"""
#         # 全角转半角 + 千分位处理
#         text = text.translate(self.num_trans)
#         text = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)  # 移除千分位逗号
#         return re.sub(r'([eE])\s*?([+-])', r'\1\2', text)  # 科学计数法规范化
        
#     def bpe(self, token: str) -> str:
#         """改进的BPE处理（继承父类并添加数字保护）"""
#         # 数字保护逻辑（网页1的token完整性思想）
#         if re.match(r'^[\d\.eE+-]+$', token):
#             return token + '</w>'
#         return super().bpe(token)
    
#     def encode(self, text: str) -> List[int]:
#         """增强编码流程"""
#         # 预处理阶段（网页1的text processing阶段）
#         text = whitespace_clean(basic_clean(text)).lower()
#         text = self._normalize_numbers(text)
        
#         # 分词编码流程
#         bpe_tokens = []
#         for token in re.findall(self.pat, text):
#             # 处理货币符号与数字组合（如$123）
#             if token.startswith(('$','€','£','¥')):
#                 num_part = token[1:]
#                 if num_part.isdigit():
#                     bpe_tokens.append(self.encoder[token[0]+'</w>'])
#                     token = num_part
            
#             # 字节编码转换（保持父类逻辑）
#             token_bytes = token.encode('utf-8')
#             token = ''.join(self.byte_encoder[b] for b in token_bytes)
            
#             # BPE处理
#             for bpe_token in self.bpe(token).split(' '):
#                 bpe_tokens.append(self.encoder[bpe_token])
#         return bpe_tokens
    
#     def decode(self, tokens: List[int]) -> str:
#         """改进的解码后处理"""
#         text = super().decode(tokens)
#         # 恢复科学计数法显示（网页1的后处理思想）
#         text = re.sub(r'(\d)\s*([eE])\s*([+-]?\d+)', r'\1\2\3', text)
#         # 恢复货币符号连接
#         return re.sub(r'(\$|€|£|¥)\s+(\d)', r'\1\2', text)

# def tokenize(_tokenizer, texts: Union[str, List[str]], 
#             context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
#     """将文本转换为模型输入张量[3,7](@ref)
#     Args:
#         texts: 输入文本（单条或多条）
#         context_length: 最大序列长度（CLIP标准为77）
#         truncate: 超长时是否截断
#     Returns:
#         shape: (batch_size, context_length) 的填充后张量
#     """
#     if isinstance(texts, str):
#         texts = [texts]

#     # 特殊标记编码
#     sot_token = _tokenizer.encoder["<|startoftext|>"]
#     eot_token = _tokenizer.encoder["<|endoftext|>"]
    
#     # 编码所有文本
#     all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] 
#                 for text in texts]
    
#     # 处理PyTorch版本差异
#     dtype = torch.long if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0") else torch.int
#     result = torch.zeros(len(all_tokens), context_length, dtype=dtype)
    
#     # 填充张量
#     for i, tokens in enumerate(all_tokens):
#         if len(tokens) > context_length:
#             if truncate:  # 截断处理
#                 tokens = tokens[:context_length]
#                 tokens[-1] = eot_token  # 确保结尾标记
#             else:
#                 raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
#         result[i, :len(tokens)] = torch.tensor(tokens)
    
#     return result.long()
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re
import os
from typing import Any, Union, List
import packaging

import torch

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model_data/bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


def tokenize(_tokenizer, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result.long()