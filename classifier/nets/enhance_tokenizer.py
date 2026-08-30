import gzip
import html
import os
from functools import lru_cache
from typing import Any, Union, List

import ftfy
import regex as re
import packaging

import torch


@lru_cache()
def default_bpe():
    """返回预训练的BPE词汇表路径，使用LRU缓存避免重复加载"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model_data/bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """创建字节到Unicode字符的映射表，
    解决BPE编码中的不可打印字符问题，返回字典{字节: Unicode字符}
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    # 处理0-255范围内的不可打印字符
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """获取连续字符对集合，用于BPE合并"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    """基础文本清洗：
    1. 修复常见文本错误（ftfy）
    2. 双重HTML转义解码
    """
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """空格规范化：
    1. 合并连续空格为单个空格
    2. 去除首尾空格
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class SimpleTokenizer(object):
    """基于BPE的简单分词器，支持特殊标记处理"""
    def __init__(self, bpe_path: str = default_bpe()):
        # 初始化字节编码器/解码器
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # 加载BPE合并规则（从 gzip 文件中读取）
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        # 注意：此处根据原代码保留第2行到特定行
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges if merge.strip() != '']

        # 构建词汇表：先加入所有字符及其带词尾标记的版本，再加入所有 BPE 合并生成的 token
        vocab = list(bytes_to_unicode().values())
        vocab += [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])  # 加入特殊标记

        # 创建编码器/解码器映射
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        # 缓存：防止重复计算，特殊标记预先加入缓存
        self.cache = {
            '<|startoftext|>': '<|startoftext|>',
            '<|endoftext|>': '<|endoftext|>'
        }
        # 修改正则表达式，使得连续数字作为一个整体匹配
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|
            [\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE | re.VERBOSE)

    def bpe(self, token):
        """执行BPE编码，将 token 处理成合并后的子 token 序列（空格分隔）"""
        if token in self.cache:
            return self.cache[token]
        
        # 注：移除了之前针对纯数字 token 的特殊处理，
        # 让所有 token 均走统一的 BPE 算法。
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            self.cache[token] = token + '</w>'
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
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
                except ValueError:
                    new_word.extend(word[i:])
                    break

                # 如果当前 pair 与待合并的匹配，则进行合并
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        
        word_str = ' '.join(word)
        self.cache[token] = word_str
        return word_str

    def encode(self, text):
        """文本编码流程：
        1. 清洗文本
        2. 正则分词
        3. 对于纯数字 token，直接逐字符映射；
           对于其它 token，先进行字节转换，再调用 BPE 编码，
           最后检查每个 BPE 子 token 是否在词汇表中，如果遇到形如合并后的纯数字 token（如 "110</w>"），
           则将其拆分成单个数字 token。
        """
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()

        for token in re.findall(self.pat, text):
            # 针对纯数字 token：逐字符映射，每个字符（最后一个字符添加 '</w>'）均已在词汇表中
            if token.isdigit():
                for i, char in enumerate(token):
                    if i == len(token) - 1:
                        digit_tok = char + '</w>'
                    else:
                        digit_tok = char
                    if digit_tok not in self.encoder:
                        raise KeyError(f"Digit token {digit_tok} not found in encoder.")
                    bpe_tokens.append(self.encoder[digit_tok])
                continue

            # 对其它 token 先进行字节编码转换
            token_bytes = token.encode('utf-8')
            token_str = ''.join(self.byte_encoder[b] for b in token_bytes)
            bpe_str = self.bpe(token_str)
            # 使用空格分割得到 BPE 子 token
            for sub_tok in bpe_str.split(' '):
                if sub_tok not in self.encoder:
                    # 如若子 token 形如合并后的数字（例如 "110</w>"），则拆分为单个字符 token
                    if sub_tok.endswith('</w>') and sub_tok[:-len('</w>')].isdigit():
                        digits = sub_tok[:-len('</w>')]
                        for j, char in enumerate(digits):
                            if j == len(digits) - 1:
                                dtok = char + '</w>'
                            else:
                                dtok = char
                            if dtok not in self.encoder:
                                raise KeyError(f"Fallback digit token {dtok} not found in encoder.")
                            bpe_tokens.append(self.encoder[dtok])
                    else:
                        raise KeyError(f"Token `{sub_tok}` not found in encoder.")
                else:
                    bpe_tokens.append(self.encoder[sub_tok])
        return bpe_tokens

    def decode(self, tokens):
        """解码过程：
        1. 通过解码器将 token 转换成字符串
        2. 处理词尾标记
        3. 用字节解码还原原始文本
        """
        text = ''.join([self.decoder[token] for token in tokens])
        text = text.replace('</w>', ' ')
        byte_data = bytearray([self.byte_decoder[c] for c in text])
        return byte_data.decode('utf-8', errors="replace")


def tokenize(_tokenizer, texts: Union[str, List[str]], 
             context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """将文本转换为模型输入张量
    Args:
        texts: 输入文本（单条或多条）
        context_length: 最大序列长度（CLIP 标准为 77）
        truncate: 如果文本过长，是否截断
    Returns:
        shape: (batch_size, context_length) 的填充张量
    """
    if isinstance(texts, str):
        texts = [texts]

    # 获取特殊开始、结束标记
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    # 处理 PyTorch 版本差异
    dtype = torch.long if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0") else torch.int
    result = torch.zeros(len(all_tokens), context_length, dtype=dtype)

    # 对不足 context_length 的序列进行填充
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result.long()