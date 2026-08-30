import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from .bert import Transformer
from .simple_tokenizer import SimpleTokenizer, tokenize
# from .enhance_tokenizer import SimpleTokenizer, tokenize
from .vit import VisionTransformer,SimpleVisionEncoder,OutVisionEncoder

class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim          = 512,
        # vision
        input_resolution   = 224,
        vision_layers      = 12,
        vision_width       = 288,
        vision_patch_size  = 32,
        # text
        context_length      = 77,
        transformer_layers  = 12,
        transformer_width   = 768,
        transformer_heads   = 12,
        vocab_size          = 49408,
        **kwargs
    ):
        super().__init__()

        self.context_length = context_length

        vision_heads    = vision_width // 64
        self.visual     = VisionTransformer(
            input_resolution    = input_resolution,
            patch_size          = vision_patch_size,
            width               = vision_width,
            layers              = vision_layers,
            heads               = vision_heads,
            output_dim          = embed_dim
        )
        # self.visual     = OutVisionEncoder(
        #     input_resolution    = input_resolution,
        #     patch_size          = vision_patch_size,
        #     width               = vision_width,
        #     # layers              = vision_layers,
        #     # heads               = vision_heads,
        #     output_dim          = embed_dim
        # )

        self.tokenizer          = SimpleTokenizer()
        # self.tokenizer          = EnhancedNumberTokenizer()
        self.transformer        = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size             = vocab_size
        self.token_embedding        = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding   = nn.Parameter(torch.empty(self.context_length, transformer_width))

        self.text_projection        = nn.Parameter(torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)
        self.ln_final               = nn.LayerNorm(transformer_width)
        self.logit_scale            = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text_features   = None

    @property
    def dtype(self):
        # 返回的是torch.float32（数据类型）
        # return self.visual.conv1.weight.dtype
        return torch.float32
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        text = tokenize(self.tokenizer, text).to(self.visual.conv1.weight.device)
        # text = enhanced_tokenize(self.tokenizer, text).to(self.visual.conv1.weight.device)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features  = self.encode_image(image)
        image_features  = image_features / image_features.norm(dim=-1, keepdim=True)
        if self.training:
            text_features = self.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            if self.text_features is None:
                text_features = self.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.text_features = text_features.detach()
            text_features = self.text_features
        logit_scale         = self.logit_scale.exp()
        logits_per_image    = logit_scale * image_features @ text_features.t()
        logits_per_text     = logits_per_image.t()

        return logits_per_image, logits_per_text
    